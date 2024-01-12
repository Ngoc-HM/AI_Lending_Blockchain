from langchain.agents import AgentType, initialize_agent
from typing import List, Optional
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import SerpAPIWrapper
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.vectorstores import Pinecone
import pinecone
from langchain.output_parsers.enum import EnumOutputParser
import datetime
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import time
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.utilities import GoogleSearchAPIWrapper
import os
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from enum import Enum
from typing_extensions import Annotated
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import requests
from langchain.tools import tool
from typing import Optional, Type
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from requests import Request, Session
import json
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
MODEL_NAME = "gpt-35-turbo-1106"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://oxalus-hust-chatbot.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "9f9bf3570cd945a68ddf27c196807862"
os.environ["LANGCHAIN_TRACING"] = "false"
llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-1106",
            model_name="gpt-35-turbo-1106"
        )
def getRespone(url, parameters = {}):
    headers = {
                'Accepts': 'application/json',
                'X-Apikey': 'QN1qnzREpjrLQ45aw5VMHGPh1UVo9Ka72XGEExYOmC3TEOCt',
                }
    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)
        return data
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)
def get_rate_deposit_borrow_rate_on_pool(token_symbol:str)-> str:
    '''Search for the borrowing interest rate (borrow rate/borrow APY/borrow rate APY)/deposit rate (deposit rate/deposit APY/deposit rate APY) of 1 token on 1 pool'''
    
    id = "aave-v2"
    context = getRespone(f"https://develop.centic.io/dev/v3/projects/lending/{id}/overview")
    res=""
    if context != "":
        for i in range(len(context["markets"])):
            item_project=context["markets"][i]
            for j in range(len(item_project["assets"])):
                item_token=item_project["assets"][j]
                if item_token["symbol"].upper() == token_symbol.upper():
                    res=item_token
                    res["supplyAPY(deposit rate/deposit APY/deposit rate APY)"] = res.pop("supplyAPY")
                    res["borrowAPY(borrow rate/borrow APY/borrow rate APY)"] = res.pop("borrowAPY")
    return res
print(get_rate_deposit_borrow_rate_on_pool("USDT"))