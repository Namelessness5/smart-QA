from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
import json

class BaseSplitter(ABC):
    @abstractmethod
    # initialization
    def load_from_config(self, config):
        pass

    def split(self, string):
        pass


class langchain_splitter(BaseSplitter):
    def __init__(self, config):
        self.load_from_config(config)

    def load_from_config(self, config):
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, string):
        return self.splitter.split_text(string)


def load_splitter(splitter_config):
    # config_s = open("default_setting.json", 'r')
    # api_config = json.load(config_s)
    # splitter_config = api_config["splitter"]
    splitter_name = splitter_config["name"]
    if splitter_name == "langchain_splitter":
        return langchain_splitter(splitter_config)
    else:
        pass