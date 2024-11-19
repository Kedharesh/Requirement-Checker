import os
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from langchain.agents import AgentType, initialize_agent
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part

os.environ["GOOGLE_API_KEY"] = "AIzaSyDJcprygJGzLmA2fGaMBKOTFBJu4_iT9vk"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "focus-ensign-437302-v1-a14f1f892574.json"
project = "focus-ensign-437302-v1"
location = "us-central1"
bucket_name = "client-user-storage"
aiplatform.init(project=project, location=location)

class DocumentInput(BaseModel):
    question: str = Field()

# Initialize Gemini 1.5 Pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

tools = []
files = [
    {
        "name": "Requirements",
        "path": "Requirements.pdf",
    },
    {
        "name": "Invoice",
        "path": "Innvoice.pdf",
    },
]

for file in files:
    loader = PyPDFLoader(file["path"])
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    retriever = FAISS.from_documents(docs, embeddings).as_retriever()

    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=file["name"],
            description=f"Extract information from {file['name']}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )

agent = initialize_agent(
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
)

# Fixed user input
user_question = "Find the differences between items names, quantities, and other information in the Requirements and Invoice documents."

# Run the agent with the fixed input
result = agent({"input": user_question})

# Print the result
print(result['output'])
