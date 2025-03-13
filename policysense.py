from fastapi import HTTPException, Request, UploadFile, File, Form, APIRouter
from google import genai
from google.genai import types
import os
import time
from typing import Dict, Optional
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

# Configuration
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# In-memory storage for chat sessions and uploaded documents
form_chat_sessions: Dict[str, any] = {}
doc_upload_chat_sessions: Dict[str, any] = {}
insurance_chat_sessions: Dict[str, any] = {}
uploaded_documents: Dict[str, any] = {}

# System prompts
FORM_MODEL_SYS_IN = """
You are an expert in extracting structured information from unstructured sentences. 
Your task is to extract personal details such as Name, Address, Age, Phone Number, Date of Birth, 
Email, Gender, and Occupation from the user's input.
If any of these fields are missing, ask follow-up questions one at a time to gather the missing information. 
Frame your questions in a conversational tone, asking directly for the missing information 
(e.g., 'What is your phone number?'). Avoid phrasing questions with specific names like 
'What is John Doe's phone number?'.
If the user does not want to provide certain details, allow them the option to reply with 'None'. 
Once all required fields are submitted, respond in JSON format with the structured information.
"""

DOC_UPLOAD_SYS_IN = """
Objective:
You are an expert insurance advisor. Your role is to analyze user-submitted policy documents and provide accurate, concise answers directly from the document, along with relevant follow-up questions to clarify or expand on the user's inquiry.

Key Behaviors:
Document Intake & Processing:
- Accept policy documents in PDF or DOC format from the user.
- If the document is scanned, use Optical Character Recognition (OCR) to extract text for processing.
- Structure the document into key sections (e.g., policy overview, terms and conditions, exclusions) to enable precise response generation.

Text Segmentation & Structuring:
- Organize the document into easily retrievable sections (e.g., headings, paragraphs, and clauses) for accurate question answering.
- Map the structure of the document to allow efficient navigation and reference during user interactions.

Expert Training & Contextual Understanding:
- Be trained to understand and interpret legal and insurance-specific terminology.
- Develop a deep understanding of the relationships between sections like coverage, terms, conditions, and exclusions to provide clear, fact-based answers.

Handling User Questions:
- Respond to user queries with concise, on-point answers, directly extracted from the relevant sections of the document.
- Avoid unnecessary details. Focus on providing only the information that answers the user's question with clarity.
- Follow each answer with a relevant follow-up question to encourage further engagement or clarification (e.g., "Would you like more details on the exclusions?" or "Do you want to see the policy's specific terms and conditions?").

Precise Answer Generation:
- Generate answers that are directly tied to the content of the uploaded document.
- Cite specific sections or summarize only the necessary information for the user to understand the answer in a clear, accessible format.
- Offer to expand or link to relevant sections if the user wants more detailed information.

Follow-Up Queries:
- Understand the user’s follow-up questions in context, providing seamless, sequential answers that build on the previous interaction.
- Adapt responses and guide users through the document's content by providing tailored follow-up suggestions.

Continuous Learning:
- Continuously learn from new policy documents, improving your ability to handle various types of policies and offer even more accurate answers in future interactions.

Note:
- Respond in Hindi only if the user requests a response in Hindi.
- Respond in English if the user requests a response in English.
"""

INSURANCE_SYS_IN = """
You are an expert insurance advisor and chatbot that provides detailed and accurate information 
about various insurance policies. Your role is to assist users with any and all questions they 
have about insurance, including but not limited to: types of insurance (health, life, auto, home, etc.), 
policy coverage details, premium calculations, claim processes, benefits, exclusions, legal terms, 
and how to choose the right insurance plan. You respond in a clear, concise, and friendly manner, 
making complex concepts easy to understand for the user. You also provide real-life examples, 
industry insights, and step-by-step guidance when necessary.
Note:
- Respond in Hindi only if the user requests a response in Hindi.
"""


# Initialize models
form_config=[
    types.GenerateContentConfig(
        max_output_tokens=8192,
        temperature=1,
        top_p=0.95,
        top_k=64,
        response_mime_type="text/plain",
        safety_settings= [
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
        ],
        system_instruction=FORM_MODEL_SYS_IN
    )
]

doc_config=[
    types.GenerateContentConfig(
        max_output_tokens=8192,
        temperature=1,
        top_p=0.95,
        top_k=64,
        response_mime_type="text/plain",
        safety_settings= [
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
        ],
        system_instruction=DOC_UPLOAD_SYS_IN
    )
]

insurance_config=[
    types.GenerateContentConfig(
        max_output_tokens=8192,
        temperature=1,
        top_p=0.95,
        top_k=64,
        response_mime_type="text/plain",
        safety_settings= [
            types.SafetySetting(
                category='HARM_CATEGORY_HATE_SPEECH',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_HARASSMENT',
                threshold='BLOCK_NONE'
            ),
            types.SafetySetting(
                category='HARM_CATEGORY_DANGEROUS_CONTENT',
                threshold='BLOCK_NONE'
            ),
        ],
        system_instruction=INSURANCE_SYS_IN
    )
]


# Utility functions
def get_or_create_session(user_id: str, sessions: Dict[str, any], config_type: list) -> any:
    """Retrieve or create a chat session for the user."""
    if user_id not in sessions:
        sessions[user_id] = client.chats.create(model="gemini-2.0-flash", config=config_type)
    return sessions[user_id]

def upload_to_gemini(path: str) -> any:
    """Upload a file to Gemini."""
    return client.files.upload(file=path)

def wait_for_files_active(files: list[any]) -> None:
    """Wait for the given files to be active."""
    for file in files:
        while file.state.name == "PROCESSING":
            time.sleep(10)
            file = client.files.get(file.name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")

# Form auto-filling bot endpoint
@router.post("/form")
async def update_form(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("user_id")
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")

        chat_session = get_or_create_session(user_id, form_chat_sessions, form_config)
        response = chat_session.send_message(query)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting structured information: {str(e)}")

# Document upload bot endpoint
@router.post("/policydoc-upload")
async def upload_policy_document(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        file_path = f"{TEMP_DIR}/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        uploaded_file = upload_to_gemini(file_path)
        wait_for_files_active([uploaded_file])

        uploaded_documents[user_id] = uploaded_file
        return {"message": "Document uploaded successfully", "file_name": uploaded_file.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post("/policydoc-chatbot")
async def continue_policy_document_chat(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        user_id = body.get("user_id")
        language = body.get("language")
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")

        if user_id not in uploaded_documents:
            raise HTTPException(status_code=400, detail="No document uploaded for this user")

        chat_session = get_or_create_session(user_id, doc_upload_chat_sessions, doc_config)
        response = chat_session.send_message([uploaded_documents[user_id], f"{query} in {language}"])
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Insurance chatbot endpoint
@router.post("/insurance")
async def insurance_chatbot(request: Request):
    try:
        body = await request.json()
        query = body.get("query")
        language = body.get("language")
        user_id = body.get("user_id")
        if not query or not user_id:
            raise HTTPException(status_code=400, detail="Query and user ID are required")

        chat_session = get_or_create_session(user_id, insurance_chat_sessions, insurance_config)
        response = chat_session.send_message(f"{query} in {language}")
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")