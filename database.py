from DBclient import *
import json


def load_database(database_name):
    config_s = open("api.json", 'r')
    api_config = json.load(config_s)
    database_config = api_config[database_name]
    if database_name == "chroma":
        return ChromaDatabaseClient(database_config)
    elif database_name == "pgvector":
        return PgVectorDatabase(database_config)
    elif database_name == "milvus":
        return MilvusDatabaseClient(database_config)
    else:
        pass
