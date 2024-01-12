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
from langchain.agents import AgentType, initialize_agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.output_parsers import PydanticOutputParser
MODEL_NAME = "gpt-35-turbo-1106"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = "https://oxalus-hust-chatbot.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "9f9bf3570cd945a68ddf27c196807862"
os.environ["LANGCHAIN_TRACING"] = "false"
list=[[['Ethereum is a cryptocurrency, the symbol is ETH, the slug is ethereum', 'iEthereum is a cryptocurrency, the symbol is IETH, the slug is iethereum'], ['Bitcoin is a cryptocurrency, the symbol is BTC, the slug is bitcoin', 'Bitcoin 21 is a cryptocurrency, the symbol is XBTC21, the slug is bitcoin-21']], [['Dark Knight is a cryptocurrency, the symbol is DKNIGHT, the slug is darkknight', 'DarkCrypto is a cryptocurrency, the symbol is DARK, the slug is darkcrypto']]]
llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-1106",
            model_name="gpt-35-turbo-1106",
            openai_api_version="2023-07-01-preview"
        )
           
template = """you are crypto chatbot.Use reference sources to find additional information for user input
            Given the user input.
            [bold]You will get SLUG of cryptocurrency from user input, remember not a symbol.[/bold]
            If you don't know its slug then there's no need to add it to the answer
            User input:
            {user_input}
            Reference:{document}
            Output: I want you just return exactly slug of cryptocurrency, not talk more anything.If there are multiple slugs, each slug is separated by a comma In case don't find slug cryptocurrency just return ""
            Output:
            """
            
prompt_template = PromptTemplate(
    input_variables=["user_input","document"],
    template=template,
)
user_input="btc price"
chain = LLMChain(llm=llm, prompt=prompt_template)
response = chain.run(
    user_input=user_input,document=list
)
print(response)
def tach_chuoi(chuoi):
    # Sử dụng hàm split để tách chuỗi theo dấu phẩy và khoảng trắng sau dấu phẩy
    danh_sach_chuoi_con = chuoi.split(', ')
    new=""
    for i in danh_sach_chuoi_con:
     if i==danh_sach_chuoi_con[-1]:
      new+=i
     else:
         new=new+i+","
    return new
print(tach_chuoi(response))