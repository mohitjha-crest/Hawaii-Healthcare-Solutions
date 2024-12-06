import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()


def get_model():
    llm = ChatOpenAI(
        model_name=os.environ.get('MODEL_NAME'),
        api_key=os.environ.get('OPENAI_API_KEY'),
    )
    return llm
