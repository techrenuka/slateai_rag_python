from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_xai import ChatXAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# Initialize ElevenLabs client with environment variable
elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY')
if not elevenlabs_api_key:
    print("Warning: ELEVENLABS_API_KEY not found in environment variables")
    client = None
else:
    client = ElevenLabs(api_key=elevenlabs_api_key)

# Initialize FastAPI app
app = FastAPI(
    title="Website Chatbot API",
    description="An API to answer questions about a website using a RAG pipeline.",
    version="1.0.0"
)

# Initialize Google Generative AI model
model = ChatXAI(model="grok-3-mini-beta")

# Add CORS middleware configuration BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, adjust for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Load pre-scraped content
def load_website_content():
    try:
        file_path = os.path.join(os.path.dirname(__file__), "newslate_scraped_content.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "\n\n".join(page["text"] for page in data)
    except Exception as e:
        raise Exception(f"Error loading scraped content: {str(e)}")

# Text processing function
def process_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

# Vector store creation
def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_documents(documents, embeddings)

# RAG pipeline setup
def setup_rag_pipeline(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    llm = model
    
    prompt = PromptTemplate(
        template="""You are SLATE AI's intelligent virtual assistant.

            Your role is to help users understand and explore the services, features, and offerings of the SLATE AI platform. Use only the information provided to you through context documents retrieved from the knowledge base. If you don’t have enough information to answer a question, politely guide the user to contact support or visit the appropriate section of the website.

            Be clear, professional, and helpful in tone. Avoid guessing or fabricating answers. Always prioritize accuracy and relevance. If the question is unrelated to SLATE AI or its services, kindly redirect the user back to relevant topics.

            Examples of what you should be able to help with:
            - Describing SLATE AI's features
            - Explaining how a specific service works
            - Assisting with basic troubleshooting or FAQs
            - Guiding users to the right place on the website

            Never reveal that you are an AI language model. Act like a real-time assistant embedded in the website.

             NOTE: If the question is not relevant to the website content, respond with:
                  "The question is not relevant to the website content."
        
        Context from website:
        {context}
        
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Initialize vector store at startup
try:
    content = load_website_content()
    documents = process_text(content)
    vector_store = create_vector_store(documents)
    rag_chain = setup_rag_pipeline(vector_store)
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")
    raise

# Endpoint to handle questions
def generate_audio_response(text):
    try:
        if not elevenlabs_api_key:
            print("Warning: ElevenLabs API key not found. Audio generation disabled.")
            return None
            
        # Configure voice settings for better quality and stability
        # voice_settings = VoiceSettings(
        #     stability=0.75,  # Higher stability for more consistent output
        #     similarity_boost=0.75,  # Better voice matching
        #     style=0.0,  # Neutral style
        #     use_speaker_boost=True  # Enhanced clarity
        # )
        
        # Get the voice instance
        # voice = client.get_voice("bella")
        
        # Generate audio using the client API
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id='21m00Tcm4TlvDq8ikWAM',
            model_id="eleven_flash_v2_5",  # Latest model for better quality
            # voice_settings=voice_settings
        )

        # Convert the generator to bytes
        audio_bytes = b''.join(chunk for chunk in audio_stream)
        
        # Generate a unique filename for the audio
        audio_filename = f"response_{hash(text)}.mp3"
        audio_path = os.path.join(os.path.dirname(__file__), "audio_responses", audio_filename)
        
        # Create audio_responses directory if it doesn't exist
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Clean up old audio files (keep only last 50 files)
        audio_dir = os.path.dirname(audio_path)
        audio_files = sorted(
            [f for f in os.listdir(audio_dir) if f.endswith('.mp3')],
            key=lambda x: os.path.getctime(os.path.join(audio_dir, x))
        )
        if len(audio_files) > 50:
            for old_file in audio_files[:-50]:
                try:
                    os.remove(os.path.join(audio_dir, old_file))
                except Exception as e:
                    print(f"Warning: Could not remove old audio file {old_file}: {str(e)}")
        
        # Save the audio file
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        return audio_filename
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        
        # Get answer using the pre-initialized RAG chain
        answer = rag_chain.invoke(question)
        
        # Generate audio response
        audio_filename = generate_audio_response(answer)
        
        response = {
            "question": question,
            "answer": answer,
            "audio_file": audio_filename
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Website Chatbot API. Use the /ask endpoint to query a website."}

# Mount the audio_responses directory for serving audio files
audio_dir = os.path.join(os.path.dirname(__file__), "audio_responses")
os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    try:
        audio_path = os.path.join(audio_dir, filename)
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)