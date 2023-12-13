import sys
sys.path.append("QA_version_2")
from load_chatmodel import load_chat_model
from load_embeddingmodel import load_embedding
from load_splitter import load_splitter
# from utils import *
from database import load_database
import os
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import json


class botv2(object):
    def __init__(self):
        self.set_defualt("default_setting.json")
        self.chat_history = ChatMessageHistory()
        self.memory_rounds = 4

        self.update_history([])

    def set_defualt(self, config_path):
        r = open(config_path, "r")
        bot_config = json.load(r)
        self.set_model(bot_config["model_name"])
        self.set_embedding(bot_config["embedding"])
        self.set_spliter(bot_config["splitter"])
        self.set_vector_database(bot_config["vector_database"])
        r.close()

    def set_model(self, model_name):
        self.model = load_chat_model(model_name)

    def set_embedding(self, embedding):
        self.embedding = load_embedding(embedding)

    def set_vector_database(self, vector_database):
        self.vector_database = load_database(vector_database)

    def set_spliter(self, spliter_config):
        self.splitter = load_splitter(spliter_config)

    def set_table(self, table_name="default_table_name"):
        self.vector_database.load_database(table_name)

    def reset_table(self):
        self.vector_database.reset_table()

    def update_vector_database_from_string(self, string: str, source="default"):
        strings = self.splitter.split(string)
        sources = [source for item in strings]
        self.vector_database.update_vector_database_from_strings(strings, self.embedding, sources)

    def update_history(self, history: list):
        self.chat_history = ChatMessageHistory()
        for item in history:
            self.chat_history.add_user_message(item[0])
            self.chat_history.add_ai_message(item[1])
        memory = ConversationBufferMemory(k=self.memory_rounds, chat_memory=self.chat_history)
        self.conversation = ConversationChain(
            llm=self.model,
            memory=memory,
            verbose=False,
            prompt=PromptTemplate(input_variables=['history', 'input'],
                                  template='\n这是一段发生在友善的人工智能和人类用户的对话，人工智能会尽可能地回答用户的问题。\n现在的对话:\n{history}\n人类: {input}\n人工智能:')

        )

    def delete(self):
        pass

    # chat with file
    def file_based_qa(self, query: str, filter: list, k=5, threshold=0):
        # return query2QA(self.qa, query)
        query_results = self.vector_database.similarity_search_with_score(query, embedding=self.embedding, k=k, filter=filter)
        reference = []
        question = "根据以下文档：\n"
        reference_id = []
        source = []
        source_id = []
        for i in range(len(query_results)):
            item = query_results[i]
            if float(item[1]) >= threshold:
                reference.append(item[0])
                question += f"\"\"\"{item[0]}\"\"\",\n"
                reference_id.append(item[2])

        question += f"回答问题:{query}"
        result = self.model.predict(question)
        for i in range(len(reference)):
            # source += f"\n原文:{reference[i]} id:{reference_id[i]} "
            source.append(reference[i])
            source_id.append(reference_id[i])
        return result, source, source_id

    # chat with model
    def chat(self, text: str):
        return self.conversation.predict(input=text), [], []
        # return self.conversation.predict(input=text)

    def react(self, query, refer=False, filter=[], k=5):
        if refer:
            return self.file_based_qa(query, filter=filter, k=k)
        else:
            return self.chat(query)

    def delete_by_id(self, ID: str):
        self.database.delete(self.database.get(where={"source": ID})["ids"])
