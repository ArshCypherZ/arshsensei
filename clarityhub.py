from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
import os
import json
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_video_link(query, max_results=1):
    """Fetches YouTube video links for a given search query."""
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    request = youtube.search().list(
        q=query, 
        part="snippet",
        maxResults=max_results,
        type="video"
    )
    response = request.execute()
    video_links = [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in response.get("items", [])]
    return video_links if video_links else ["No video found"]

# Models
class InputData(BaseModel):
    syllabus: str
    subject: str
    class_level: str 
    exam: str

class Topic(BaseModel):
    name: str
    subtopics: List[str]
    completion_time: int  # Days to complete
    resources: List[str]
    youtube_link: str

class Input(BaseModel):
    topic: str

class Output(BaseModel):
    answer: str

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str

class QuizRequest(BaseModel):
    topic: str
    num_questions: int = 5  # Default 5 MCQs

# Routes
@router.post("/generate_topics/", response_model=List[Topic])
async def generate_topics(input_data: InputData):
    try:
        prompt = f"""
        List topics for the following syllabus in JSON format.

        Syllabus: {input_data.syllabus}
        
        Subject: {input_data.subject}
        Class Level: {input_data.class_level}
        Exam: {input_data.exam}

        Use this JSON schema:
        
        Topic = {{'name': str, 'subtopics': list[str], 'completion_time': int, 'resources': list[str]}}
        Return: list[Topic]

        name: Name of the topic
        subtopics: List of subtopics for the topic
        completion_time: Number of days to complete the topic
        resources: List of resources for the topic.

        Note: For resources, mention recommended resources for each topic.
        """
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        topic_list = json.loads(response.text.strip("```json").strip("```"))
        for topic in topic_list:
            topic["youtube_link"] = get_video_link(topic["name"])[0]
        return topic_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/about-topic/", response_model=List[Output])
async def about_topic(inp: Input):
    try:
        prompt = f"""
        Generate 100-200 words summary about {inp.topic}.
        Provide output in JSON format using the schema: {{"answer": "your_summary_here"}}
        """
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        data = json.loads(response.text.strip("```json").strip("```"))
        return [Output(**data)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_quizzes/", response_model=List[QuizQuestion])
async def generate_quizzes(quiz_request: QuizRequest):
    try:
        prompt = f"""
        Generate {quiz_request.num_questions} multiple-choice questions (MCQs) for the topic: {quiz_request.topic}.
        Provide output in JSON format using the schema: QuizQuestion = {{'question': str, 'options': list[str], 'answer': str}}
        Each question should have four options, and the answer should be one of the options.
        """
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return json.loads(response.text.strip("```json").strip("```"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
