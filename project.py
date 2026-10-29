import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# 1. Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)


# 2. Company policy documents
documents = [
    "Refund requests are allowed within 7 days.",
    "Python classes happen Monday to Friday.",
    "Students get 3 interview opportunities.",
    "Course access is available for 6 months."
]


# 3. Get user's question
print("----- Eva: Your AI Chatbot -----")
question = input("What is your query?\n").strip()

if not question:
    raise ValueError("Please enter a question.")


# 4. Create document embeddings
document_vectors = []

for document in documents:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=document
    )

    document_vectors.append(result.embeddings[0].values)

document_vectors = np.array(document_vectors)


# 5. Create question embedding
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question
)

question_vector = np.array(result.embeddings[0].values)


# 6. Calculate cosine similarity
similarities = (
    document_vectors @ question_vector
    / (
        np.linalg.norm(document_vectors, axis=1)
        * np.linalg.norm(question_vector)
    )
)

# Find the most relevant document
best_index = int(np.argmax(similarities))
best_document = documents[best_index]


# 7. Send retrieved context to the LLM
prompt = f"""
Answer the question using only the context below.

Context:
{best_document}

Question:
{question}

Instructions:
- Give a clear and concise answer.
- Do not make up information.
- If the context does not contain the answer, say:
  "I don't have enough information to answer this question."
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


# 8. Display the final answer
print("\nEva's Answer:")
print(response.text)