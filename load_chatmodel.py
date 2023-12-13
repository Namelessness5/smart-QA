import json
from langchain.chat_models import ChatOpenAI
# set the apis according to https://github.com/xusenlinzy/api-for-open-llm and write them in api.json


def load_chat_model(model_name):
    config_s = open("api.json", 'r')
    api_config = json.load(config_s)
    model_config = api_config[model_name]
    if model_config["openai_api_base"] == "default":
        return ChatOpenAI(openai_api_key=model_config["openai_api_key"], model_name=model_config["model_name"])
    else:
        return ChatOpenAI(openai_api_key=model_config["openai_api_key"],
                          openai_api_base=model_config["openai_api_base"],
                          model_name=model_config["model_name"])


if __name__ == "__main__":
    model = load_chat_model("baichuan-13b-chat")
    print(model.predict("你好"))
