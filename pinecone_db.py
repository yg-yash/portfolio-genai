import json
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
import torch
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINE_CONE_API_KEY"))

# Create a Pinecone index
index_name = "bio-data-index"
if index_name not in pc.list_indexes():
    pc.create_index(index_name, dimension=768, metric="cosine", spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ))  # Dimension should match your embedding size
index = pc.Index(index_name)

# Load the JSON data
with open("bio.json", "r", encoding="utf-8") as f:
    bio_data = json.load(f)

# Extract relevant fields
docs = []

# Personal Details
docs.append(f"Name: {bio_data['name']}")
docs.append(f"Title: {bio_data['title']}")
docs.append(f"Location: {bio_data['location']}")
docs.append(f"Contact Email: {bio_data['contact']['email']}")
docs.append(f"LinkedIn: {bio_data['contact']['linkedin']}")
docs.append(f"Website: {bio_data['contact']['website']}")

# Experience
for exp in bio_data["experience"]:
    docs.append(f"Company: {exp['company']}, Role: {exp['title']}, Location: {exp['location']}, From: {exp['start_date']} to {exp['end_date']}")
    docs.extend(exp["responsibilities"])
    docs.extend(exp["achievements"])

# Education
for edu in bio_data["education"]:
    docs.append(f"Degree: {edu['degree']} at {edu['institution']} ({edu['start_date']} - {edu['end_date']})")

# Skills
for category, skills in bio_data["skills"].items():
    docs.append(f"{category.capitalize()} Skills: {', '.join(skills)}")

# Projects
for project in bio_data["portfolio"]["projects"]:
    docs.append(f"Project: {project['title']} - {project['about']}")
    docs.extend(project["key_features"])
    docs.append(f"Technologies: {', '.join(project['technologies_used'])}")

# Hugging Face model for embeddings
model_name = "facebook/bart-base"  # Choose your preferred Hugging Face model

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings using Hugging Face model
def get_embeddings_huggingface(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling for sentence embeddings
    return embeddings

# Get embeddings for your docs using Hugging Face model
embeddings = get_embeddings_huggingface(docs)

# Prepare Pinecone data
pinecone_data = [
    {
        "id": f"doc_{i}",
        "values": embedding.tolist(),  # Convert numpy array to list for Pinecone
        "metadata": {"text": doc}
    }
    for i, (doc, embedding) in enumerate(zip(docs, embeddings))
]

# Upsert data into Pinecone
index.upsert(vectors=pinecone_data)

# Save the metadata (optional)
chroma_data = {
    "ids": [f"doc_{i}" for i in range(len(docs))],
    "embeddings": embeddings.tolist(),  # Convert numpy array to list for saving
    "metadatas": [{"text": doc} for doc in docs]
}

# Save as a .json file (optional)
output_file = "pinecone_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chroma_data, f, indent=4)

print(f"✅ Pinecone-compatible data saved to {output_file}")
