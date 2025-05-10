from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_xai import ChatXAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

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
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Load pre-scraped content
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
        template="""You are an expert assistant for the website https://newslate.co.uk/. Your task is to extract and present accurate, concise information strictly based on the content available on this website. Do not use external knowledge or assumptions. If the information is not available on the website, respond with:
        "The answer is not available in the provided website content."
        Response Format Rules (must be followed strictly):
        - Use section headings in bold (e.g., **AI Assistant Development**).
        - Use clean bullet points only (no paragraphs, no inline lists)
        - Do not include any introductory or closing text
        - Do not repeat information under different headings
        - Maintain a professional and minimal tone

        Output Example:

        AI Assistant Development
        - Design and build AI assistants to automate tasks and enhance user interaction
        - Includes NLP, context awareness, multi-modal interaction, and learning systems

        AI Strategy and Consulting
        - Strategic AI roadmaps and tech assessments
        - Implementation planning and ROI optimization
        - Risk management and compliance review
        
        PoC Development
        - Build and test AI Proof-of-Concepts
        - Swift prototyping and user-driven iteration
        - Feasibility and scalability evaluation

        Optimizing AI
        - Audit and fix failed AI implementations
        - Optimize models for better accuracy and speed
        - Revive and upgrade abandoned AI projects

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
@app.post("/ask", response_model=dict)
async def ask_question(request: QuestionRequest):
    try:
        question = request.question
        
        # Get answer using the pre-initialized RAG chain
        answer = rag_chain.invoke(question)
        
        return {"question": question, "answer": answer}
        print(answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Website Chatbot API. Use the /ask endpoint to query a website."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)