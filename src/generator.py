import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

class GeminiFlashGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        print("Gemini 1.5 Flash initialized.")

    def refine_query(self, query):
        # Uses Gemini to rephrase vague or informal queries for clarity
        prompt = f"Rephrase the following question to be more clear and specific without changing its meaning:\n\n{query}"
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return query  # Fallback: return original if rephrasing fails

    def generate(self, query, context_chunks):
        # Trim context if too long
        MAX_TOKENS = 3000
        context = '\n'.join(context_chunks)
        if len(context) > MAX_TOKENS:
            context = context[:MAX_TOKENS] + "\n[Content truncated due to length]"

        prompt = (
            "You are a helpful assistant that answers questions strictly based on the context provided below.\n"
            "Do not use outside knowledge. If the answer is not found, respond with:\n"
            "'The information is not available in the provided documents.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {e}"
