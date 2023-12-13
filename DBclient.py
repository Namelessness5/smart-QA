import socket
import pickle
from abc import ABC, abstractmethod
from pgvector.psycopg import register_vector
import psycopg
import requests
import json
from pymilvus import CollectionSchema, FieldSchema, DataType, connections, db, Collection, utility


class Database(ABC):
    @abstractmethod
    # update current database from a list of string
    def update_vector_database_from_strings(self, strings: list, embedding, metadata):
        pass

    # load a database from existing database
    def load_database(self, database):
        pass

    # search the documents with filter in the format of list
    def similarity_search_with_score(self, query: str, embedding, filter: list, k=5):
        pass

    def delete_by_id(self, source: str):
        pass

class ChromaDatabaseClient(Database):
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.default_table_name = config['default_table_name']
        self.table_name = config['default_table_name']
        self.url = f"http://{self.host}:{self.port}"
        self.reset_table()
        self.start_id = 0

    def reset_table(self):
        url = f'{self.url}/set'
        data = {'reset': True, 'table_name': self.table_name}
        response = requests.post(url, json=data)
        return response.json()

    def update_vector_database_from_strings(self, files: list, embedding, sources: list):
        response = embedding.client.embeddings.create(input=files, model=embedding.name)
        embeddings = [v.embedding for v in response.data]
        sources = [{"source": s} for s in sources]

        url = f'{self.url}/store'
        data = {'embeddings': embeddings, 'sources': sources, "files": files, "start_id": self.start_id, "table_name": self.table_name}
        self.start_id += len(files)
        response = requests.post(url, json=data)
        return response.json()

    def load_database(self, database):
        url = f'{self.url}/set'
        if database == "default_table_name":
            database = self.default_table_name
        self.table_name = database
        data = {'reset': False, 'table_name': database}
        response = requests.post(url, json=data)
        return response.json()

    def similarity_search_with_score(self, query: str, embedding, filter: list, k=5):
        response = embedding.client.embeddings.create(input=query, model=embedding.name)
        query_embedding = response.data[0].embedding
        url = f'{self.url}/get'
        query_data = {'embedding': query_embedding, 'filter': filter, 'k': k, "table_name": self.table_name}
        result = requests.get(url, json=query_data)
        result = result.content
        result = json.loads(result.decode('utf-8'))
        result_content = result['documents'][0]
        result_distance = result['distances'][0]
        result_distance = list(map(lambda x: 1-x, result_distance))
        result_source = result['metadatas'][0]
        result_source = list(map(lambda x: x['source'], result_source))
        return list(zip(result_content, result_distance, result_source))

class PgVectorDatabase(Database):
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.dbname = config['dbname']
        self.user = config['user']
        self.password = config['password']
        self.connection = psycopg.connect(dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port)
        self.connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
        register_vector(self.connection)
        self.default_table_name = config['default_table_name']
        self.table_name = config['default_table_name']
        self.reset_table()

    def reset_table(self):
        table_name = self.table_name
        self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        # self.connection.execute(f'CREATE TABLE {table_name} (id bigserial PRIMARY KEY, content text, embedding vector({embedding_length}), source text)')
        self.connection.execute(
            f'CREATE TABLE {table_name} (id bigserial PRIMARY KEY, content text, embedding vector, source text)')

    def load_database(self, database="default_table_name"):
        if database == "default_table_name":
            database = self.default_table_name
        self.table_name = database
        self.connection.execute(
            f'CREATE TABLE {database} (id bigserial PRIMARY KEY, content text, embedding vector, source text) IF NOT EXISTS {database}')


    def similarity_search_with_score(self, query: str, embedding, filter: list, k=5):
        response = embedding.client.embeddings.create(input=query, model=embedding.name)
        query_embedding = response.data[0].embedding
        neighbors = self.connection.execute(
            'SELECT (content, 1 - (embedding <=> \'%s\'), source) FROM %s WHERE source IN %s ORDER BY embedding <=> \'%s\' > 0 LIMIT %s' % (query_embedding, self.table_name, tuple(filter), query_embedding, k)).fetchall()
        # print(neighbors[0])
        neighbors = [list(item[0]) for item in neighbors]
        return neighbors

    def update_vector_database_from_strings(self, files: list, embedding, sources: list):
        response = embedding.client.embeddings.create(input=files, model=embedding.name)
        embeddings = [v.embedding for v in response.data]
        for content, embedding, source in zip(files, embeddings, sources):
            self.connection.execute(f'INSERT INTO {self.table_name}' +' (content, embedding, source) VALUES (%s, %s, %s)', (content, embedding, source))

    def delete_by_id(self, source: str):
        pass

class MilvusDatabaseClient(Database):
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.default_table_name = config['default_table_name']
        self.table_name = config['default_table_name']
        self.conn = connections.connect(
            host=self.host,
            port=self.port,
            db_name="default"
        )
        self.reset = True
        self.count_id = 1
        self.reset_table()

    def set_schema(self, embedding_length):
        file_id = FieldSchema(
            name="file_id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        content = FieldSchema(
            name="content",
            dtype=DataType.VARCHAR,
            max_length=65535,
        )
        source = FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=65535,
            # The default value will be used if this field is left empty during data inserts or upserts.
            # The data type of `default_value` must be the same as that specified in `dtype`.
            default_value="Unknown"
        )
        embedding = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=embedding_length
        )
        self.schema = CollectionSchema(
            fields=[file_id, content, source, embedding],
            description="Test book search",
            enable_dynamic_field=True
        )
    def reset_table(self):
        utility.drop_collection(self.table_name)
        self.reset = True

    def update_vector_database_from_strings(self, files: list, embedding, sources: list):
        response = embedding.client.embeddings.create(input=files, model=embedding.name)
        embeddings = [v.embedding for v in response.data]
        if self.reset:
            embedding_length = len(embeddings[0])
            self.set_schema(embedding_length)
            self.collection = Collection(
                name=self.table_name,
                schema=self.schema,
                using='default'
            )
            self.reset = False
        count_ids = [self.count_id + i for i in range(len(files))]
        self.count_id += len(files)
        self.collection.insert([count_ids, files, sources, embeddings])

        self.collection.drop_index(index_name="embedding")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 2}
        }

        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        utility.index_building_progress(self.table_name)

    def load_database(self, database):
        if database == "default_table_name":
            database = self.default_table_name
        self.table_name = database
        if utility.has_collection(database):
            self.collection = Collection(database)
        else:
            self.reset = True

    # search the documents with filter in the format of list
    def similarity_search_with_score(self, query: str, embedding, filter: list, k=5):
        response = embedding.client.embeddings.create(input=query, model=embedding.name)
        query_embedding = response.data[0].embedding

        search_params = {
            "metric_type": "L2",
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }

        expr = f"source in {filter}"
        self.collection.load()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            # the sum of `offset` in `param` and `limit`
            # should be less than 16384.
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=['content', 'source']
        )
        self.collection.release()
        result = [[item.content, 1 - item.distance, item.source] for item in results[0]]
        return result

    def delete_by_id(self, ID: str):
        pass