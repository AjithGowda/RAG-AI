import streamlit as st
import requests
import os
from document_loader import db

st.title("RAG AI Chat Bot")
st.warning("This is a demo of a RAG integrated chat bot. Please enter your query below and click 'Send' to get a response from the bot.")

API_URL = "http://localhost:8000/rag/invoke"

if "messages" not in st.session_state:
    st.session_state.messages = []

#display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Please enter your query")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    #send the data to llm and get the response
    with st.spinner("Generating response..."):
        try:
            # Retrieve relevant documents from vector store
            retrieved_docs = db.similarity_search(user_input, k=3)
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            
            api_response = requests.post(API_URL, json={"input": {"context": context, "question": user_input, "answer": ""}})
            if api_response.status_code != 200:
                st.error(f"Error: {api_response.status_code} - {api_response.text}")
                response = "Sorry, I couldn't generate a response."
            else:
                api_data = api_response.json()
                output = api_data.get("output", "Sorry, I couldn't generate a response.")
                # Extract the text content from the AIMessage object
                if isinstance(output, dict) and "content" in output:
                    response = output["content"]
                else:
                    response = output
        except Exception as e:
            st.error(f"Error: {str(e)}")
            response = "Sorry, I couldn't generate a response."
    #store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

