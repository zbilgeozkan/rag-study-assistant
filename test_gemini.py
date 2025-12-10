import os
from dotenv import load_dotenv
import google.generativeai as genai

# load .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in environment/.env")

genai.configure(api_key=api_key)

model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
print("Testing model:", model_name)

model = genai.GenerativeModel(model_name)

response = model.generate_content("Say a one-sentence greeting in English.")
print("Response text:", response.text)
