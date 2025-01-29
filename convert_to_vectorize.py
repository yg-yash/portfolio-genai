import json

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

# Convert to ChromaDB format
chroma_data = {
    "ids": [f"doc_{i}" for i in range(len(docs))],  # Unique IDs
    "embeddings": [],  # Placeholder for embeddings
    "metadatas": [{"text": doc} for doc in docs]  # Store original text as metadata
}

# Save as a .json file
output_file = "chroma_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chroma_data, f, indent=4)

print(f"✅ ChromaDB-compatible data saved to {output_file}")
