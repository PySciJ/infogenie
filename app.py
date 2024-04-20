import streamlit as st
from src.component.sql_querier import get_few_shot_db_chain 
from src.component.video_summarizer import extract_transcript_details, google_gemini_text_summarization
from src.component.chat import get_response
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="InfoGenie", page_icon="👀")

# infogenie_tab, fininsights_tab, clipnotes_tab, dbquery_tab = st.tabs(["InfoGenie", "FinInsights", "VidSum", "DBQuery"])




st.sidebar.title("News Article URLs 📰")


with st.sidebar:
        # Create a tab
        tab = st.selectbox("Select Tool", ["InfoGenie", "FinInsights", "ClipNotes", "DBQuery"])
        
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
        
vectordb_file_path = "faiss_index_hf.pkl"
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Url {i+1}: ")
    urls.append(url)
    
            

process_url_clicked = st.sidebar.button("Process URL")
show_vector_db_clicked = st.sidebar.button("Show vectordb")
document_add_remove_string = st.sidebar.text_input("Document to add/remove:")

st.sidebar.empty()
remove_document_clicked = st.sidebar.button("Remove doc")
add_document_clicked = st.sidebar.button("Add doc")




    


    



    
