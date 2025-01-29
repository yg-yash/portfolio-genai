from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load Pinecone environment variables
pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
index = pc.Index(host=os.getenv("PINE_CONE_HOST"))

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Pydantic model for input data
class Query(BaseModel):
    query: str

# Function to retrieve relevant context from Pinecone
def retrieve_info(query):
    # Generate query embedding
    query_embedding = embedding_model.encode(query).tolist()
    print(f"Query Embedding: {query_embedding}")
    
    # Query Pinecone for the top 3 most similar documents
    query_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    print(f"Query Results: {query_response}")
    
    if query_response['matches']:
        # Extract relevant context from the metadata for each result
        context = [match['metadata']['text'] for match in query_response['matches']]
        return " ".join(context)  # Return all context as a single string
    else:
        return "Sorry, I don't have that information."

# Function to interact with the chatbot
def chat_with_bot(user_query):
    # Retrieve relevant context based on the user's query
    context = retrieve_info(user_query)
    print("Context:", context)
    
    # Construct the chatbot prompt
    prompt = f"User: {user_query}\nAI: Based on the information I have, {context}"

    # Set pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Generate a response using the GPT-2 model
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,  # Generate a longer response
        temperature=0.6,  # Control the randomness (lower is more deterministic)
        top_p=0.9,  # Nucleus sampling
        repetition_penalty=1.3,  # Avoid repetitive responses
        do_sample=True,  # Sampling is set to true for more varied responses
        top_k=50,  # Top-k sampling to limit the search space
        pad_token_id=tokenizer.eos_token_id,  # Use the eos_token for padding
        no_repeat_ngram_size=2,  # Avoid repeating n-grams
        decoder_start_token_id=tokenizer.pad_token_id,  # Decoder start token
        early_stopping=False,  # Don't stop early, allow full response generation
    )
    
    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Define API endpoint
@app.post("/chat/")
async def chat(query: Query):
    response = chat_with_bot(query.query)
    return {"response": response}

