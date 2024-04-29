import streamlit as st
import pandas as pd
import time
from src.component.sql_querier import get_few_shot_db_chain 
from src.component.video_summarizer import extract_transcript_details, google_gemini_text_summarization
from src.component.chat import get_response
from langchain_core.messages import HumanMessage, AIMessage
from src.component.article_extractor import (load_url, 
                                             split_docs, 
                                             create_save_return_vector_store, 
                                             handle_userinput,
                                             populate_vector_store,
                                             display_vector_store)

st.set_page_config(page_title="InfoGenie", page_icon="👀")


# infogenie_tab, fininsights_tab, clipnotes_tab, dbquery_tab = st.tabs(["InfoGenie", "FinInsights", "VidSum", "DBQuery"])

st.sidebar.title("Tool Configs ⚙")


with st.sidebar:
        # Create a tab
        tab = st.selectbox("Select Tool", ["InfoGenie", "FinInsights", "ClipNotes", "DBQuery", "PDFQuery"])


url = st.sidebar.text_input("Url: ")
add_url= st.sidebar.button("add URL")
process_url_clicked = st.sidebar.button("Process URL")

show_vector_db_clicked = st.sidebar.button("Show vectordb")
document_add_remove_string = st.sidebar.text_input("Document to add/remove:")

st.sidebar.empty()
remove_document_clicked = st.sidebar.button("Remove doc")
add_document_clicked = st.sidebar.button("Add doc")


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

if tab == "FinInsights":
    st.title("FinInsights 📈")
    
    print(st.session_state)
        
    if "urls" not in st.session_state:
        st.session_state.urls = []
        
    if "chat_history_fi" not in st.session_state:
        st.session_state.chat_history_fi = []
    
    if "display_vector_store" not in st.session_state:
        st.session_state.display_vector_store = []
    
    
        
    print(st.session_state)
    if add_url:
        if url:
            st.session_state.urls.append(url)
            with st.popover("Show Urls"):
                st.markdown(st.session_state.urls)
            print(st.session_state)       
            
    
    if process_url_clicked:
        
        with st.status("Processing URLs...", expanded=True) as status:
            
            #Load data
            st.write("Data Loading...Started...✅✅✅")
            docs = load_url(st.session_state.urls)
            
            #Split data into chunks
            st.write("Text Splitting...Started...✅✅✅")
            chunks = split_docs(docs)
            
            #Create embeddings for chunks and store in vectorDB(FAISS, Chroma)
            st.write("Embedding Vector Started Building...✅✅✅")
            if "vector_store_fi" not in st.session_state:
                st.session_state.vector_store_fi = create_save_return_vector_store(chunks)     
            
            status.update(label="Processing Completed!", state="complete",expanded=False)
            
            #Check if st.session_state.chat_history_fi = [] empty list
            if not st.session_state.chat_history_fi:
                st.session_state.chat_history_fi = [
                    AIMessage(content="Hello, I am a bot. How can I help you?"),
                ]
            
            if not st.session_state.display_vector_store:
                st.session_state.display_vector_store = [1]
        
    if len(st.session_state.display_vector_store) == 1:
        store_df = populate_vector_store(st.session_state.vector_store_fi)
        display_vector_store(store_df)
                             
    for message in st.session_state.chat_history_fi:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
                     

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        if len(st.session_state.chat_history_fi) == 1:
            st.session_state.chat_history_fi.pop()
            
        response = handle_userinput(user_query)
        st.session_state.chat_history_fi.append(HumanMessage(content=user_query))
        # with st.chat_message("Human"):
        #         st.markdown(user_query)
        # with st.chat_message("AI"):
        #         st.markdown(response)
        st.session_state.chat_history_fi.append(AIMessage(content=response))

                    
    # if show_vector_db_clicked:
    #     v_dict = st.session_state.vector_store_fi.docstore._dict
    #     #v_dict = {"aab6aebf-3db4": "Document(page_content="Skip Navigation..", metadata={'source': 'www.xyz.com'})"}
        
    #     data_rows = []
    #     doc_set = set()
    #     for chunk_id in v_dict.keys():
    #         doc_name = v_dict[chunk_id].metadata["source"]
    #         doc_set.add(doc_name)
    #     for url in doc_set:
    #         data_rows.append({"documents": url})
    #         #data_rows.append({"chunk_id": chunk_id, "document": doc_name})
    #     vector_store_fi_df = pd.DataFrame(data_rows)
    #     vector_store_fi_df.index += 1
    #     display_vector_store(vector_store_fi_df)
            
   

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
        
if tab == "DBQuery":
    st.title("DBQuery 🔍")
    question = st.text_input("Question: ")
    
    if question:
        chain = get_few_shot_db_chain()
        response = chain.run(question)

        st.header("Answer")
        st.write(response)

if tab == "PDFQuery":
    st.title("PDFQuery 📑")
        



    





    


    



    
