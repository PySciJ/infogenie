import streamlit as st
from src.component.langchain_helper import get_few_shot_db_chain 
from src.component.video_summarizer import extract_transcript_details, google_gemini_text_summarization


st.title("InfoGenie👀")

question = st.text_input("Question: ")
youtube_link = st.text_input("Enter Youtube Video Link: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)

if youtube_link:
    video_id = youtube_link.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    
if st.button("Get Video Summary"):
    transcript = extract_transcript_details(youtube_link)
    summary = google_gemini_text_summarization(transcript)
    
    st.markdown("## 🧾Summary Content:")
    st.write(summary)
    
