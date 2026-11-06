
import ollama

# Test the local LLM
response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "Válaszolj magyarul: Mi az a RAG rendszer?"
        }
    ]
)

print("LLM response:")
print(response["message"]["content"])

# Test the embeddings
embedding = ollama.embed(
    model="bge-m3",
    input="Az OTP Diákhitel Számla számlavezetési díja."
)

print("\nEmbedding test:")
print("Vector dimensions:", len(embedding["embeddings"][0]))