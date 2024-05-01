import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv
from src.component.sql_querier import get_few_shot_db_chain 
from src.component.video_summarizer import extract_transcript_details, google_gemini_text_summarization
from src.component.chat import get_response
from langchain_core.messages import HumanMessage, AIMessage
from src.component.article_extractor import (load_url, 
                                             split_docs, 
                                             create_save_return_vector_store,
                                             merge_save_return_vector_store,
                                             remove_doc_update_vector_store, 
                                             handle_userinput,
                                             populate_vector_store,
                                             display_vector_store,
                                             stream_userinput)

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="InfoGenie", page_icon="👀")
st.sidebar.title("Tool Configs ⚙")

with st.sidebar:
    tab = st.selectbox("Select Tool", ["InfoGenie", "FinInsights", "ClipNotes", "DBQuery"])



### InfoGenie ###

if tab == "InfoGenie":
    st.title("InfoGenie👀")
    user_query = st.chat_input("Your question")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Conversation
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else: 
            with st.chat_message("AI"):
                st.markdown(message.content)
                
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
            
        st.session_state.chat_history.append(AIMessage(ai_response))


### FinInsights ###

if tab == "FinInsights":
    st.title("FinInsights 📈")
    
    add_url = False   
    remove_url = False

    if "remove_url" not in st.session_state:
        st.session_state.remove_url = []
        
    url = st.sidebar.text_input("Url: ")
    with st.sidebar:
        selection_remove = st.sidebar.toggle('Remove Url', value=False)

        if selection_remove:
            remove_url = st.sidebar.button("Remove doc")
            st.session_state.remove_url = [1]
            
        else:
            add_url = st.sidebar.button("Add doc")
            st.session_state.remove_url = []
        
    process_url_clicked = st.sidebar.button("Process URL")
    
    print(st.session_state)
        
    if "urls" not in st.session_state:# Temp url list as inputs to load_url()
        st.session_state.urls = []
    
    if "urls_history" not in st.session_state:# This is to track the urls in the vectordb
        st.session_state.urls_history = []
        
    if "chat_history_fi" not in st.session_state:
        st.session_state.chat_history_fi = []
    
    if "vector_store_fi" not in st.session_state:
        st.session_state.vector_store_fi = []

    print(st.session_state)
    
    if add_url:
        if url and url not in st.session_state.urls_history:
            st.session_state.urls.append(url)
            st.session_state.urls_history.append(url)
        with st.popover("Show Urls"):
            st.markdown(st.session_state.urls)
            
    if remove_url:
        if url and url in st.session_state.urls_history and url not in st.session_state.urls: #Only remove if url as been added to store
            st.session_state.urls.append(url)
            st.session_state.urls_history.append(url)
        with st.popover("Show Urls"):
            st.markdown(st.session_state.urls)
            print(st.session_state)
        
    
    if process_url_clicked:
        if len(st.session_state.remove_url) == 1:
            with st.status("Removing document...", expanded=True) as status:
                
                st.session_state.vector_store_fi = remove_doc_update_vector_store(st.session_state.urls, st.session_state.vector_store_fi)
                status.update(label="Documents Removed!", state="complete", expanded=False)               
                st.session_state.urls = [] # Clear temp urls list
                
        elif not st.session_state.remove_url:
            with st.status("Processing URLs...", expanded=True) as status:
                
                #Load data
                st.write("Data Loading...Started...✅✅✅")
                docs = load_url(st.session_state.urls)
                
                #Split data into chunks
                st.write("Text Splitting...Started...✅✅✅")
                chunks = split_docs(docs)
                
                #Create embeddings for chunks and store in vectorDB(FAISS, Chroma)
                st.write("Embedding Vector Started Building...✅✅✅")
                
                if not st.session_state.vector_store_fi: #Check if empty and initialize for the first time
                    st.session_state.vector_store_fi = create_save_return_vector_store(chunks)
                    
                
                else: # vector_store_fi already exist, merge new documents
                    st.write("Updating Vector Database...✅✅✅")
                    st.session_state.vector_store_fi = merge_save_return_vector_store(chunks, st.session_state.vector_store_fi)
                        
                    
                status.update(label="Processing Completed!", state="complete", expanded=False)
                
                st.session_state.urls = [] # Clear temp urls list after vector_store has been initialized
        
            
                
    if not isinstance(st.session_state.vector_store_fi, list): # If vector_store is not empty
        store_df = populate_vector_store(st.session_state.vector_store_fi)
        display_vector_store(store_df)
                        
    for message in st.session_state.chat_history_fi:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="🤗"):
                st.write(message.content)        
                        
                        
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        if len(st.session_state.chat_history_fi) == 1:
            st.session_state.chat_history_fi.pop()
        
        with st.chat_message("Human", avatar="🤗"):
            st.markdown(user_query)
            
        response = handle_userinput(user_query)
        st.session_state.chat_history_fi.append(HumanMessage(content=user_query))
        st.session_state.chat_history_fi.append(AIMessage(content=response))
        
        with st.chat_message("AI", avatar="🤖"):
            st.write_stream(stream_userinput(response))
    

### ClipNotes ###

if tab == "ClipNotes":
    st.title("ClipNotes 🎞")
    youtube_link = st.text_input("Enter Youtube Video Link: ")
    
    if youtube_link:
        video_id = youtube_link.split("=")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        
    if st.button("Get Video Summary"):
        transcript = extract_transcript_details(youtube_link)
        summary = google_gemini_text_summarization(transcript)
        
        st.markdown("## 🧾Summary Content:")
        st.write(summary)
        
### DBQuery ###        
        
if tab == "DBQuery":
    st.title("DBQuery 🔍")
    
    
    if "chat_history_db" not in st.session_state:
        st.session_state.chat_history_db = []
    
        
    for message in st.session_state.chat_history_db:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="🤗"):
                st.write(message.content)  
    
    question = st.chat_input("Question")
    if question is not None and question != "":
        
        with st.chat_message("Human", avatar="🤗"):
            st.markdown(question)
            
        chain = get_few_shot_db_chain()
        response = chain.run(question)
        st.session_state.chat_history_db.append(HumanMessage(content=question))
        st.session_state.chat_history_db.append(AIMessage(content=response))
        
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(response)


        



    





    


    



    
