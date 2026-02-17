import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = PdfReader("data/sample.pdf")
data_path = os.path.join(project_root, "data", data)
chroma_db_path = os.path.join(project_root, "chroma_db")

text_loader = TextLoader(data_path)
documents = text_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3")
embeddings = OllamaEmbeddings(model=MODEL_NAME)

#vector store
db = Chroma.from_documents(texts, embeddings, persist_directory=chroma_db_path)

print("Document loaded and vector store created successfully.")