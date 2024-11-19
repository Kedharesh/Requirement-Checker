import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.agents import AgentType, initialize_agent
import tempfile
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

os.environ["GOOGLE_API_KEY"] = "AIzaSyDJcprygJGzLmA2fGaMBKOTFBJu4_iT9vk"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
project = "focus-ensign-437302-v1"
location = "us-central1"
bucket_name = "client-user-storage"
aiplatform.init(project=project, location=location)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DocumentInput(BaseModel):
    question: str = Field(...)

def process_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()
    return retriever

@app.post("/compare-documents/")
async def compare_documents(
    requirements_file: UploadFile = File(...),
    invoice_file: UploadFile = File(...),
    question: str = "Find the differences between items names, quantities, and other information in the documents."
):
    # Create temporary files for uploaded PDFs
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as requirements_temp, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as invoice_temp:
        
        requirements_temp.write(await requirements_file.read())
        invoice_temp.write(await invoice_file.read())
        requirements_temp.close()
        invoice_temp.close()

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    # Process documents and create tools
    tools = []
    files = [
        {"name": "Requirements", "path": requirements_temp.name},
        {"name": "Invoice", "path": invoice_temp.name},
    ]

    for file in files:
        retriever = process_document(file["path"])
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"Extract information from {file['name']}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
        )

    # Initialize agent
    agent = initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
    )

    # Run agent with question
    try:
        result = agent({"input": question})
        output = result['output']
    except Exception as e:
        output = f"An error occurred: {str(e)}"

    # Clean up temporary files
    os.unlink(requirements_temp.name)
    os.unlink(invoice_temp.name)

    return JSONResponse(content={"result": output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)