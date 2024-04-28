import streamlit as st
from src.component.sql_querier import get_few_shot_db_chain 
from src.component.video_summarizer import extract_transcript_details, google_gemini_text_summarization
from src.component.chat import get_response
from langchain_core.messages import HumanMessage, AIMessage
from src.component.article_extractor import load_url, split_docs, create_save_return_vector_store, handle_userinput

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
    # main_placeholder = st.empty()
    print(st.session_state)
        
    if "urls" not in st.session_state:
        st.session_state.urls = []
    print(st.session_state)
    if add_url:
        if url:
            st.session_state.urls.append(url)
            with st.popover("Show Urls"):
                st.markdown(st.session_state.urls)
            print(st.session_state)        
    # user_question = st.text_input("Question about your articles")
    # if user_question:
    #     if st.session_state.conversation is None:
    #         st.session_state.conversation = get_conversation_chain()

    #     response = st.session_state.conversation({'question': user_question})
    #     st.write(response)
    #     st.session_state.chat_history = response['chat_history']


    
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

            if "chat_history_fi" not in st.session_state:
                st.session_state.chat_history_fi = [
                    AIMessage(content="Hello, I am a bot. How can I help you?"),
                ]
                             
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
        st.session_state.chat_history_fi.append(AIMessage(content=response))
            
        for message in st.session_state.chat_history_fi:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content) 
   

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
        



    





    


    



    
