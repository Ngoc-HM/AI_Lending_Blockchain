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
class structure_output(BaseModel):
    generated_questions:Optional[List[str]]=Field(description=" if no question generated,return "" ")
parser = PydanticOutputParser(pydantic_object=structure_output)
template = """


As a crypto chatbot, my task is to identify knowledge gaps from user input and create questions for unknown terms. Here's a concise guide:
Follow step-by-step instructions:
- Step 1: Separate Input:

You MUST separate user input into terms  (e.g., user input:"give me the borrow rate of Wrapped BTC on aave v2 " .separate into terms:  "Wrapped BTC", "aave v2").
- Step 2: Knowledge Check:

Assess if each term is known in the blockchain context,by thinking about and answering the questions yourself (e.g."do I know aave v2?").[bold]This step MUST be very precise [/bold].Ask yourself many times to do it correctly
- Step 3: Define Unknown Terms:

If a term is [bold]UNFAMILIAR[/bold], generate a question about its definition (Each Unknown term creates 1 question)in the blockchain context (e.g., "what is ...in the blockchain context").

***
FOR EXAMPLE:
user input: I would like to know the borrow rate of Wrapped BTC in marginfi Lending?
Step 1: Separate Input:
- Terms:  "Wrapped BTC" ,"marginfi Lending"

Step 2: Knowledge Check:
- Do I know Wrapped BTC? Yes
- Do I know marginfi Lending? No

Step 3: Define Unknown Terms:
- Generate a question about radiant-v2: "What is marginfi Lending in the blockchain context?"

***

Begin!
user input: {user_input}
output:
 """
            
prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=template,
    
)
user_input=" hello"
chain = LLMChain(llm=llm, prompt=prompt_template)
response = chain.run(
    user_input=user_input
)
#response=llm.predict("do you know Wrapped BTC  ")
print(response)
import re

def extract_questions(text):
    # Step 1: Extract terms
    terms_match = re.search(r'Terms: "(.*?)"', text)
    if not terms_match:
        return []

    terms = [term.strip() for term in terms_match.group(1).split('","')]

    # Step 2: Knowledge Check
    knowledge_check = re.findall(r'Do I know (.*?)\? (Yes|No)', text)

    # Step 3: Define Unknown Terms
    unknown_terms = [term[0] for term in knowledge_check if term[1] == 'No']
    questions = [f'What is {term} in the blockchain context?' for term in unknown_terms]

    return questions
print(extract_questions(response))
