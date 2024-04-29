import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import os
load_dotenv()

def extract_transcript_details(video_url):
    try:
        video_id = video_url.split("=")[1] # http:/www.youtube.watch?v=HFfXvfFe9F8
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        # transcript_text[0] = {'text': 'hello all my name is crush naak and', 'start': 0.599, 'duration': 3.841}
        
        transcript = ""
        for i in transcript_text: 
            transcript += "" + i['text'] # extract only the text portion of the transcript
        
        return transcript
    except Exception as e:
        raise e

def google_gemini_text_summarization(transcript_text):
    summarization_prompt = """Extract key points from the video transcript and summarize the main ideas and concepts discussed. 
    Break down the main ideas and provide a few key points under that main idea in point form.
    Focus on capturing important insights, trends, and findings relevant to the topic covered in the video, regardless of the domain. 
    Provide the summary of the text given here:  """

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(summarization_prompt+transcript_text)
    return response.text
