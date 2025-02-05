import re
import replicate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore, storage

load_dotenv()

# Initialize Firebase Admin SDK
cred = credentials.Certificate("./firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-project-id.appspot.com'  # Replace with your Firebase bucket name
})

# Firestore and Storage Clients
db = firestore.client()
bucket = storage.bucket()


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set "*" to allow all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Initialize the replicate client
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


class Prompt(BaseModel):
    name: str
    band: str
    year: int


class GenerateRequest(BaseModel):
    prompt: Prompt


class Metadata(BaseModel):
    generated_text: str
    photo_url: str
    capitalized_words_count: int
    words_followed_by_numbers_count: int
    year_parity: str
    name: str
    band: str       #
    year: int


@app.post("/generate_text/")
async def generate_text(request: GenerateRequest):
    try:
        prompt = (
            f"Write exactly two paragraphs about the band {request.prompt.name}. \n\n"
            f"Incorporate the user's explanation of why they like this band: \"{request.prompt.band}\". \n\n"
            f"The writing style can be any, as long as it demonstrates basic generative capabilities. \n\n"
            f"Make sure to mention the chosen year: {request.prompt.year}."
        )

        input_data = {
            "prompt": prompt,
            "temperature": 0.75,
            "max_new_tokens": 800
        }

        generated_text = ""
        for event in replicate_client.stream("meta/llama-2-7b-chat", input=input_data):
            if event and event.data:
                generated_text += event.data  # Extract generated text

        # 1. Count of Words That Start with a Capital Letter
        capitalized_words_count = len([word for word in generated_text.split() if word[0].isupper()])

        # 2. Count of Words Followed by Numbers
        words_followed_by_numbers_count = len([word for word in generated_text.split() if re.search(r'\d$', word)])

        # 3. Check if the selected year is odd or even
        year_parity = "Odd" if request.prompt.year % 2 != 0 else "Even"

        return {"data": {
            "capitalized_words_count": capitalized_words_count,
            "words_followed_by_numbers_count": words_followed_by_numbers_count,
            "year_parity": year_parity,
            "generated_text": generated_text.strip()
        }}

    except replicate.exceptions.ReplicateError as e:
        return {"error": str(e)}


@app.post("/generate_photo/")
async def generate_photo(request: GenerateRequest):
    try:
        # Build the prompt for the image generation
        prompt = (
            f"Generate a photo of a band name - {request.prompt.name}. \n\n"
            f"Incorporate the user's explanation of why they like this band: \"{request.prompt.band}\". \n\n"
            f"Make sure to get photo of the chosen year: {request.prompt.year}."
        )

        input_data = {"prompt": prompt, "scheduler": "K_EULER"}

        # Generate the image with the given prompt
        output = replicate_client.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input=input_data
        )

        # Save the image and store the file path
        image_paths = []
        for index, item in enumerate(output):
            image_path = f"output_{index}.png"
            with open(image_path, "wb") as file:
                file.write(item.read())
            image_paths.append(image_path)

        # Return the image paths in the response
        return {"data": image_paths[0]}

    except Exception as e:
        return {"error": str(e)}


@app.post("/store_metadata/")
async def store_metadata(metadata: Metadata):
    try:
        # Store the data in Firestore
        data = {
            "generated_text": metadata.generated_text,
            "photo_url": metadata.photo_url,
            "capitalized_words_count": metadata.capitalized_words_count,
            "words_followed_by_numbers_count": metadata.words_followed_by_numbers_count,
            "year_parity": metadata.year_parity,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "name": metadata.name,
            "band": metadata.band,
            "year": metadata.year,
        }

        # Add the data to a Firestore collection
        doc_ref = db.collection("generated_data").add(data)

        # Return the document ID
        return {"document_id": doc_ref[1].id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing metadata: {str(e)}")


@app.get("/get_last_metadata/", response_model=Metadata)
async def get_last_metadata():
    try:
        # Query the most recent entry from Firestore (sorted by creation time)
        docs = db.collection("generated_data") \
            .order_by("timestamp", direction=firestore.Query.DESCENDING) \
            .limit(1).stream()

        # Get the first (and only) document
        doc = next(docs, None)

        if not doc:
            raise HTTPException(status_code=404, detail="No document found")

        # Convert Firestore document to dictionary
        doc_data = doc.to_dict()

        # Return the data as a Metadata object
        return Metadata(
            generated_text=doc_data.get("generated_text"),
            photo_url=doc_data.get("photo_url"),
            capitalized_words_count=doc_data.get("capitalized_words_count"),
            words_followed_by_numbers_count=doc_data.get("words_followed_by_numbers_count"),
            year_parity=doc_data.get("year_parity"),
            name=doc_data.get("name"),
            band=doc_data.get("band"),
            year=doc_data.get("year"),
        )

    except Exception as e:
        print(f"Error fetching data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

