# from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

embedding = OllamaEmbeddings(model="llama3")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding) 


llm_prompt = """
                You are a helpful assistant that answers questions based on the following retrieved documents
                Context: {context}
                Question: {question}
                Answer:
            """

prompt = ChatPromptTemplate.from_messages([
    ("system", llm_prompt), ("human", "{context} {question} {answer}")])



# llm = OllamaLLM(model="llama3", temperature=0.5)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

chain  = prompt | llm