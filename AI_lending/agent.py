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
llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-1106",
            model_name="gpt-35-turbo-1106"
        )
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="1a100")
memory = ConversationBufferMemory(
    chat_memory=message_history)
chat_history = memory.buffer
question="WBTC price"

#question="borrow rate of USDT at aave-v2"
class SearchInput(BaseModel):
        token_symbol: str = Field(description="symbol of token cryptocurrency,eg BTC")
class SearchInput2(BaseModel):
        full_question: str = Field(description="full question input")

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

def getLendingPoolName(user_input):
    template = """Given the user input
    You will get full name of lending pool in order to ask for TVL (total value locked), total deposit, or total borrow of a lending pool (some pool numbers will not have information). 
    For example: user input "with Venus lending pool" you will get Venus, or "in Venus" you will get Venus too"
    User input:
    {user_input}

    Output: I want you just return exactly lending pool name, not talk more anything. In case don't find lending pool name just return ""
    Output:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        user_input=user_input
    )
    return response
def get_symbol_token(user_input):
    template = """you are crypto chatbot.
    Given the user input.
    You will get symbol token cryptocurrency.
    For example: user input "Bitcoin Price" you will return "BTC"
    User input:
    {user_input}

    Output: I want you just return exactly symbol token, not talk more anything. In case don't find symbol token just return ""
    Output:
    """
    
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        user_input=user_input
    )
    return response
def compare_strings_ignore_spaces_and_hyphens(str1, str2):
    # Loại bỏ các ký tự không mong muốn (dấu cách và dấu trừ) từ cả hai chuỗi
    cleaned_str1 =''.join(char for char in str1 if char not in [' ', '-'])
    cleaned_str2 =''.join(char for char in str2 if char not in [' ', '-'])

    # So sánh chuỗi đã làm sạch
    return cleaned_str1.lower() == cleaned_str2.lower()
def getPoolId(lending_pool_name):
    res = getRespone("https://develop.centic.io/dev/v3/ranking/defi?category=Lending")
    for doc in res["docs"]:
        if(compare_strings_ignore_spaces_and_hyphens(doc["name"],lending_pool_name)):
            return doc["id"]
    return ""
def response_token_price(info_tokens):
    """ response_schemas = [
        ResponseSchema(name="token name", description="full name of token"),
        ResponseSchema(
            name="price",
            description="price of token"),
        ResponseSchema(name="chain name", description="name of chain")
    ]"""
    class response_schemas(BaseModel):
     token_name: str = Field(description="full name of token")
     price: str = Field(description="price of token")
     chain_name: str = Field(description="name of chain")
    parser = PydanticOutputParser(pydantic_object=response_schemas)
    #output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    #format_instructions = output_parser.get_format_instructions()
    promptprice = PromptTemplate(
        template="returns all provided token information in the format provided.\n output:{format_instructions}\n context:{info_token}",
        input_variables=["info_token"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    list_res=[]
    chain  = LLMChain(llm=llm, prompt=promptprice)
    #for i in info_token:
    response=chain.run({"info_token":info_tokens})
    #list_res.append[response]
    return response

def get_chain_name(chainID):
    res= getRespone("https://develop.centic.io/dev/v3/common/chains")
    for id,chain in res["chains"].items():
     if (chainID==chain["id"]):
        chain_name=chain["name"]
    return chain_name
def get_structure_price(symbol_token):
    res=getRespone(f"https://develop.centic.io/dev/v3/tokens/price?symbols={symbol_token}")
    list_token=[]
    for token in res["tokens"]:
     token["name_chain"]=get_chain_name(token["chainId"])
     token["full_name_token"]=token.pop("name")
     token["price(USD)"]=token.pop("price")
     list_token.append(token)
    return list_token
     
@tool
def getPoolInfo(user_input: str) -> str:
    """Ask for TVL (total value locked), total deposit or total supply, or total borrow of a lending pool (some pool numbers will not have information)"""
    lending_pool_name = getLendingPoolName(question)
    id = getPoolId(lending_pool_name)
    context = getRespone(f"https://develop.centic.io/dev/v3/projects/lending/{id}/overview")
    if context != "":
        context["markets"] = ""
    return context
def chuyen_doi_ve_BTC(dau_vao):
    if ':' in dau_vao:
        _, token_symbol = dau_vao.split(':')
        return token_symbol
    else:
        return dau_vao

@tool(args_schema=SearchInput2)
def get_rate_deposit_borrow_rate_on_pool(full_question:str)-> str:
    '''Search for the borrowing interest rate (borrow rate/borrow APY/borrow rate APY)/deposit rate (deposit rate/deposit APY/deposit rate APY) of 1 token on 1 pool'''
    
    token_symbol=get_symbol_token(question)
    lending_pool_name = getLendingPoolName(question)
    id = getPoolId(lending_pool_name)
    context = getRespone(f"https://develop.centic.io/dev/v3/projects/lending/{id}/overview")
    res=""
    if context != "":
        for i in range(len(context["markets"])):
            item_project=context["markets"][i]
            for j in range(len(item_project["assets"])):
                item_token=item_project["assets"][j]
                if item_token["symbol"].upper() == token_symbol.upper():
                    res=item_token
                    res["(%)supplyAPY(deposit rate/deposit APY/deposit rate APY)"] = res.pop("supplyAPY")
                    res["(%)borrowAPY(borrow rate/borrow APY/borrow rate APY)"] = res.pop("borrowAPY")
    return res
def get_price(token_symbol):
            url = 'https://develop.centic.io/dev/v3/tokens/price'
            parameters = {
            'symbols':token_symbol.upper()
            
            
            }
            headers = {
            'Accepts': 'application/json',
            'X-Apikey': 'QN1qnzREpjrLQ45aw5VMHGPh1UVo9Ka72XGEExYOmC3TEOCt',
            }

            session = Session()
            session.headers.update(headers)

            try:
                response = session.get(url, params=parameters)
                data = json.loads(response.text)
            except (ConnectionError, Timeout, TooManyRedirects) as e:
                print(e)
            return data

@tool(args_schema=SearchInput)
def get_token_price(token_symbol:str):
            """Search for token price by symbol token, return all token"""
            result=[]
            result=get_structure_price(get_symbol_token(question))
            res=response_token_price(result)
            return res
        
        

            
tools = [get_token_price,get_rate_deposit_borrow_rate_on_pool,getPoolInfo]
prefix = """You are a crypto chatbot. Answer the following questions as best you can. You have access to the following tools:"""
suffix = """Begin! if use tool "get_token_price", you must return all data
        
        
        Question: {input}
        
        
        {agent_scratchpad}
        """

def refined_coin_question(user_question):
    template= """rephrase the follow-up question to be a standalone question
    If user input contains cryptocurrency abbreviations, you must replace it to the full name. For example, 'BTC' must be replaced into 'Bitcoin' 
    question: {input}

    refined question:
    
    """   
    prompt_template = PromptTemplate(
        input_variables = ["input"],
            template = template,
        )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    refined_coin_question = chain.run(
        
            input=user_question
            
        )
    return refined_coin_question
prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad",]
)
tool_names = [tool.name for tool in tools]
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True,handle_parsing_errors=True
)

#efined_coin_question=refined_coin_question(user_question)
#response = agent_executor.run(input=refined_coin_question)

agent_executor.run(input=question)
#name=getLendingPoolName(question)
#print(name)
#print(getPoolId(name))



