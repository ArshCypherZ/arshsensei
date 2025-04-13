from typing import List, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
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

# Enums for quiz types
class DifficultyLevel(str, Enum):
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"

class QuestionType(str, Enum):
    multiple_choice = "multiple-choice"
    true_false = "true-false"
    mixed = "mixed"

# Models
class InputData(BaseModel):
    syllabus: str
    subject: str
    class_level: str 
    exam: str
    difficulty: str
    timeline: str
    priorKnowledge: str

class Topic(BaseModel):
    name: str
    subtopics: List[str]
    completion_time: int
    resources: List[str]
    youtube_link: str

class Input(BaseModel):
    topic: str

class Output(BaseModel):
    answer: str

class MCQQuestion(BaseModel):
    question: str
    options: List[str] = Field(..., min_length=2)
    answer: str

class TrueFalseQuestion(BaseModel):
    question: str
    options: List[str] = ["True", "False"]
    answer: str

QuizQuestionResponse = Union[MCQQuestion, TrueFalseQuestion]

class QuizRequest(BaseModel):
    topic: str
    difficulty: DifficultyLevel
    questionCount: int = Field(default=5, gt=0)
    questionType: QuestionType

class Flashcard(BaseModel):
    question: str
    hint: str
    answer: str

class FlashcardRequest(BaseModel):
    topic: str
    difficulty: str
    num_flashcards: int

class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]
    error: str = None
    message: str = None

# Routes
@router.post("/generate_topics/", response_model=List[Topic])
async def generate_topics(input_data: InputData):
    try:
        prompt = f"""
        Generate a detailed list of topics based on the following requirements, provided in JSON format.

        Syllabus: {input_data.syllabus}
        Subject: {input_data.subject}
        Class Level: {input_data.class_level}
        Target Exam: {input_data.exam}
        Difficulty Level: {input_data.difficulty}
        Desired Timeline: {input_data.timeline}
        Prior Knowledge: {input_data.priorKnowledge}

        Provide output in this format:
        [
            {{
                "name": "Topic Name",
                "subtopics": ["sub1", "sub2"],
                "completion_time": 5,
                "resources": ["Resource A", "Resource B"]
            }}
        ]
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        topic_list = json.loads(response.text.strip("```json").strip("```").strip())

        for topic in topic_list:
            topic["youtube_link"] = (get_video_link(topic["name"]))[0]

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

@router.post("/generate_quizzes/", response_model=List[QuizQuestionResponse])
async def generate_quizzes(quiz_request: QuizRequest):
    try:
        prompt = f"""
        Generate {quiz_request.questionCount} quiz questions about the topic: "{quiz_request.topic}".
        The difficulty level should be {quiz_request.difficulty.value}.
        The question type should be {quiz_request.questionType.value}.
        """

        if quiz_request.questionType == QuestionType.multiple_choice:
            prompt += """
        For multiple-choice questions, provide output as a JSON list using this schema:
        {{
            "question": "string",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A"
        }}
        """
        elif quiz_request.questionType == QuestionType.true_false:
            prompt += """
        For true/false questions, provide output as a JSON list using this schema:
        {{
            "question": "string",
            "options": ["True", "False"],
            "answer": "True" or "False"
        }}
        """
        elif quiz_request.questionType == QuestionType.mixed:
            prompt += f"""
        Provide a mix of both multiple-choice and true/false questions (approx. half of each).
        Output should be a JSON list using:
        - MCQ: {{ "question": "string", "options": ["A", "B", "C", "D"], "answer": "string" }}
        - True/False: {{ "question": "string", "options": ["True", "False"], "answer": "True" or "False" }}
        """

        prompt += "\nReturn ONLY the JSON list, no markdown or explanations."

        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        json_string = response.text.strip("```json").strip("```").strip()

        try:
            quiz_data = json.loads(json_string)
            if not isinstance(quiz_data, list):
                raise ValueError("Response is not a JSON list")

            validated_list = []
            for item in quiz_data:
                if "options" in item and len(item["options"]) > 2:
                    validated_list.append(MCQQuestion(**item))
                elif "options" in item and item["options"] == ["True", "False"]:
                    validated_list.append(TrueFalseQuestion(**item))
                else:
                    raise ValueError(f"Unrecognized question format: {item}")

            return validated_list

        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON from Gemini")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error validating questions: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quizzes: {str(e)}")

@router.post("/generate_flashcards", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest):
    try:
        prompt = f"""
        Generate {request.num_flashcards} flashcards on the topic '{request.topic}' with difficulty '{request.difficulty}'.
        Guidelines:
        - Use active recall
        - Phrase questions to provoke thinking
        - Mix subtopics
        - Keep them short and clear
        - Use Cloze Deletion if helpful

        Provide output as:
        {{
            "flashcards": [
                {{
                    "question": "...",
                    "hint": "...",
                    "answer": "..."
                }}
            ]
        }}
        """

        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

        flashcards_data = json.loads(response.text.strip("```json").strip("```").strip())
        if "flashcards" not in flashcards_data:
            raise HTTPException(status_code=500, detail="Missing flashcards in response")

        flashcards = [
            Flashcard(**fc)
            for fc in flashcards_data['flashcards']
        ]

        return FlashcardResponse(flashcards=flashcards, message="Flashcards generated successfully.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
