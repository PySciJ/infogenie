import pickle
import time
import pandas as pd
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain_google_vertexai import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#Load URLS
def load_url(urls):
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    return docs

def split_docs(docs):
    r_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ",", "."],
        chunk_size=2000,
        chunk_overlap=500
    )
    chunks = r_splitter.split_documents(docs)
    
    return chunks

def create_save_return_vector_store(chunks):
    vectordb_file_path = "faiss_index_hf.pkl"
    instructor_embeddings = HuggingFaceInstructEmbeddings()
    vector_store = FAISS.from_documents(documents=chunks, embedding=instructor_embeddings)
    time.sleep(1)
    # with open(vectordb_file_path, "wb") as f:
    #     pickle.dump(vector_db, f)
    # with open(vectordb_file_path, "rb") as f:
    #     vector_store = pickle.load(f)
    
    vector_store.save_local(vectordb_file_path)     
    return vector_store

def merge_save_return_vector_store(chunks, vector_store):
    vectordb_file_path = "faiss_index_hf.pkl"
    instructor_embeddings = HuggingFaceInstructEmbeddings()
    extension_db = FAISS.from_documents(documents=chunks, embedding=instructor_embeddings)
    time.sleep(1)
    vector_store.merge_from(extension_db)
    
    # with open(vectordb_file_path, "wb") as f: # Overwrite embeddings pickle file
    #     pickle.dump(existing_vector_store, f)
    
    # with open(vectordb_file_path, "rb") as f:
    #     vector_store = pickle.load(f)
    vector_store.save_local(vectordb_file_path)
    
    return vector_store
            
def get_context_retriever_chain(vector_store): # Retrieves relevant documents to chat_history and current user question
    # chat_model = ChatVertexAI(model="text-bison@001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # vectordb_file_path = "faiss_index_hf.pkl"
    # # instructor_embeddings = HuggingFaceInstructEmbeddings()
    # # vector_store = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)
    
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=chat_model,
    #     retriever=vector_store.as_retriever(search_type="mmr"),
    #     memory=memory
    # )
    # return conversation_chain
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history_fi"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(chat_model, vector_store.as_retriever(), prompt)
    
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    
    # SystemMessage: Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history_fi"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(chat_model, prompt) # Create a chain that includes the context(relevant documents)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def handle_userinput(user_query):
    
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store_fi)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history_fi": st.session_state.chat_history_fi,
        "input": user_query
    })
    
    return response['answer']

def stream_userinput(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)



def populate_vector_store(vector_store_fi):
    v_dict = vector_store_fi.docstore._dict
    #v_dict = {"aab6aebf-3db4": "Document(page_content="Skip Navigation..", metadata={'source': 'www.xyz.com'})"}
    
    data_rows = []
    doc_set = set()
    for chunk_id in v_dict.keys():
        doc_name = v_dict[chunk_id].metadata["source"]
        doc_set.add(doc_name)
    for url in doc_set:
        data_rows.append({"documents": url})
        #data_rows.append({"chunk_id": chunk_id, "document": doc_name})
    vector_store_fi_df = pd.DataFrame(data_rows)
    vector_store_fi_df.index += 1
    return vector_store_fi_df
    
    
def display_vector_store(store_df):
    with st.expander("Documents Database"):
        st.write(store_df)
        
