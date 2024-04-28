from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_response(query, chat_history):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro")
    
    prompt_template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:
    Chat history: {chat_history}
    
    User question: {user_question}
    
    If the history of the conversation does not provide any information about User's question, answer based on your knowledge.
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    
    chain = prompt | chat_model | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })