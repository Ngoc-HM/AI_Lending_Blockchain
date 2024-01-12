from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain.agents import AgentType, initialize_agent
from requests import Session
import json
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor


MODEL_NAME = "gpt-35-turbo-1106"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oxalus-hust-chatbot.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "9f9bf3570cd945a68ddf27c196807862"
os.environ["LANGCHAIN_TRACING"] = "false"
CHAT_API_VERSION = "2023-07-01-preview"

llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo-1106",
            model_name="gpt-35-turbo-1106",
            openai_api_version=CHAT_API_VERSION
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

def getLendingPoolName(user_input):
    template = """Given the user input
    You will get full name of lending pool in order to ask for TVL (total value locked), total deposit, or total borrow of a lending pool (some pool numbers will not have information). 
    For example: user input "with Venus lending pool" you will get Venus, or "in Venus" you will get Venus too"
    User input:
    {user_input}

    Output: I want you just return exactly lending pool name, not talk more anything. In case don't find lending pool name just return ""
    Standalone question:
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

def getPoolId(lending_pool_name):
    res = getRespone("https://develop.centic.io/dev/v3/ranking/defi?category=Lending")
    for doc in res["docs"]:
        if(doc["name"].lower() == lending_pool_name.lower()):
            return doc["id"]
    return ""

class SearchInput(BaseModel):
    coin_name: str = Field(
        description="Ask for TVL (total value locked), total deposit and total supply, and total borrow of a lending pool (some pool numbers will not have information)")
            
@tool(args_schema=SearchInput)
def getPoolInfo(user_input: str) -> str:
    """Ask for TVL (total value locked), total deposit and total supply, and total borrow of a lending pool (some pool numbers will not have information)"""
    lending_pool_name = getLendingPoolName(user_input)
    id = getPoolId(lending_pool_name)
    context = getRespone(f"https://develop.centic.io/dev/v3/projects/lending/{id}/overview")
    context["market"] = ""

    print(f"context: {context}")
    return context

tools = [getPoolInfo]

agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
        )
input = "give me total value, total deposit, total borrow locked in JustLend"
response = agent.run(input)
print(response)
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
# )
# response = agent.run(user_question)
