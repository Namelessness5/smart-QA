import json
from openai import OpenAI
# set the apis according to https://github.com/xusenlinzy/api-for-open-llm and write them in api.json


class my_embedding(object):
    def __init__(self, config, name):
        if config["openai_api_base"] == "default":
            self.client = OpenAI(api_key=config["openai_api_key"])
        else:
            self.client = OpenAI(api_key=config["openai_api_key"], base_url=config["openai_api_base"])
        self.name = name


def load_embedding(model_name):
    config_s = open("api.json", 'r')
    api_config = json.load(config_s)
    model_config = api_config[model_name]
    return my_embedding(model_config, model_config["model_name"])

    # if model_config["openai_api_base"] == "default":
    #     return OpenAIEmbeddings(openai_api_key=model_config["openai_api_key"], model=model_config["model_name"])
    # else:
    #     return OpenAIEmbeddings(openai_api_key=model_config["openai_api_key"],
    #                             openai_api_base=model_config["openai_api_base"],
    #                             model_name=model_config["model_name"])
