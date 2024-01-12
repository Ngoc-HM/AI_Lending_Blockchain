from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import uvicorn
from chat import Chatbot

# from Demo_Chatbot import Chatbot
app = FastAPI()

class UserQuestion(BaseModel):
    question: str
   




@app.post("/chat", status_code=200)
async def chat(user_question: UserQuestion):
    chatbot_crypto =Chatbot()
    try:
        final_answer=""
        relative= ""
        #final_answer = chatbot_crypto.chat(user_question.question)
        intent = chatbot_crypto.get_intent(user_question.question)
        print(intent)
        if "communication" in intent:
            final_answer=chatbot_crypto.communication(user_question.question)
            #relative = rcm().final(user_question.question)
        if "other" in intent:
            final_answer=chatbot_crypto.search_api(user_question.question)
            #relative = rcm().final(user_question.question)

        #return {"answer": final_answer, "relative": relative}
        return {"answer":final_answer}
    except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))
        return {"error": f"{e}"}