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
        chunk_size=1000,
        chunk_overlap=200
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

def remove_doc_update_vector_store(urls, vector_store):
    vectordb_file_path = "faiss_index_hf.pkl"
    v_dict = vector_store.docstore._dict
    data_rows = []
    for chunk_id in v_dict.keys():
        doc_name = v_dict[chunk_id].metadata["source"]
        data_rows.append({"chunk_id": chunk_id, "document": doc_name})
    vector_store_df_with_chunk_id = pd.DataFrame(data_rows)
    st.write(vector_store_df_with_chunk_id)
    conditions = []
    for url in urls:
        conditions.append(vector_store_df_with_chunk_id['document'] == url)

    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition |= condition #comb_cond= comb_cond | cond

        # Apply the combined condition to filter `vector_df`
    filtered_vector_store_df = vector_store_df_with_chunk_id[combined_condition]
    chunk_list = filtered_vector_store_df["chunk_id"].tolist()
    #with st.container():
        #st.write(chunk_list)
    vector_store.delete(chunk_list)
    time.sleep(1)
    vector_store.save_local(vectordb_file_path)

    
    return vector_store

def get_context_retriever_chain(vector_store): # Retrieves relevant documents to chat_history and current user question
   
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    retriever = vector_store.as_retriever(search_type="mmr")
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history_fi"),
        ("human", "{input}"),
        # ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # Create a chain that takes conversation history and returns documents.
    history_aware_retriever = create_history_aware_retriever(chat_model, retriever, contextualize_q_prompt)
    
    return history_aware_retriever


def get_conversational_rag_chain(history_aware_retriever):
    chat_model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
        
    {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", qa_system_prompt),  # SystemMessage: Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
      MessagesPlaceholder(variable_name="chat_history_fi"),
      ("human", "{input}"),
    ])
   
    stuff_documents_chain = create_stuff_documents_chain(chat_model, prompt) # Create a chain that includes the context(relevant documents)
                                                                    # prompt: Prompt template. Must contain input variable "context", which will be used for passing in the formatted documents.
    return create_retrieval_chain(history_aware_retriever, stuff_documents_chain)


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
        
