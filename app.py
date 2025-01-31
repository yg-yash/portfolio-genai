from fastapi import FastAPI
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
import os
from pinecone import Pinecone

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load Pinecone environment variables
pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))
index = pc.Index(host=os.getenv("PINE_CONE_HOST"))

# Hugging Face Inference API URLs and Token
HF_API_URL_EMBEDDINGS = "https://api-inference.huggingface.co/models/facebook/bart-base"  # Replace with your sentence transformer model URL
#HF_API_URL_EMBEDDINGS = "https://api-inference.huggingface.co/models/sentence-transformers/all-distilroberta-v1"  # Replace with your sentence transformer model URL
HF_API_URL_GPT2 = "https://api-inference.huggingface.co/models/gpt2"  # Replace with your model's URL
HF_API_TOKEN = os.getenv("HUGGING_FACE_API_KEY")

# Pydantic model for input data
class Query(BaseModel):
    query: str

# Function to retrieve relevant context from Pinecone
def retrieve_info(query):
    # Generate query embedding
    query_embedding = get_embedding_from_huggingface(query)
    print(f"Query Embedding: {query_embedding}")
    # query_vector = np.array(query_embedding)
    # # Use PCA to reduce to 384 dimensions
    # pca = PCA(n_components=384, svd_solver='auto')
    # reduced_vector = pca.fit_transform(query_vector.reshape(1, -1))  # Reshaping to 2D (1, 768) for PCA

    
    # # Query Pinecone for the top 3 most similar documents
    query_response = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    
    # print(f"Query Results: {query_response}")
    
    if query_response['matches']:
        # Extract relevant context from the metadata for each result
        context = [match['metadata']['text'] for match in query_response['matches']]
        return " ".join(context)  # Return all context as a single string
    else:
        return "Sorry, I don't have that information."


# Function to get sentence embedding from Hugging Face model
def get_embedding_from_huggingface(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
    }
    payload = {"inputs": text}
    print("headers",headers,payload)
    response = requests.post(HF_API_URL_EMBEDDINGS, headers=headers, json=payload)
    print(f"Status Code: {response.status_code}")
    # print(f"Response Content: {response.json()}")
    print("Response Text:", response.json()[0][0])

    if response.status_code == 200:
        # Assuming the API returns the embeddings in response
        return  response.json()[0][0]
    else:
        return []  # Return empty if there is an error

# Function to interact with Hugging Face's GPT-2 model
def chat_with_huggingface(query, context):
    # Construct the prompt with user query and context
    prompt = f"User: {query}\nAI: Based on the information I have, {context}"

    # Make a request to Hugging Face API
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
    }
    payload = {"inputs": prompt}
    
    # Send the request to Hugging Face API
    response = requests.post(HF_API_URL_GPT2, headers=headers, json=payload)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Sorry, I couldn't generate a response."

# Define API endpoint
@app.post("/chat/")
async def chat(query: Query):
    #Retrieve relevant context based on the user's query
    context = retrieve_info(query.query)
    print("Context:", context)

    # # Get the response from Hugging Face model
    response = chat_with_huggingface(query.query, context)
    
    return {"response": response}

