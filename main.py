from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env
 
import httpx

app = FastAPI()

class CommentRequest(BaseModel):
    comment: str

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

@app.post("/classify")
async def classify_comment(request: CommentRequest):
    prompt_text = (
        f"Is the following comment positive or negative? "
        f"Comment: '{request.comment}' "
        "Answer with either 'Good' or 'Bad'."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GEMINI_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    candidates = data.get("candidates", [])
    if not candidates:
        return {"classification": "Unknown"}

    output = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().lower()

    if "good" in output:
        classification = "Good"
    elif "bad" in output:
        classification = "Bad"
    else:
        classification = "Unknown"

    return {"classification": classification}
