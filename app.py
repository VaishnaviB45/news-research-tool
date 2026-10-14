import os
import time
import streamlit as st
from google import genai
from google.genai.errors import ServerError
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")
urls = [
    st.sidebar.text_input(f"URL {i+1}", key=f"sidebar_url_{i+1}") 
    for i in range(3)
]
process_url_clicked = st.sidebar.button("Process URLs", key="btn_process_urls")

file_path = "faiss_store_index"
main_placeholder = st.empty()

# 1. Initialize Google GenAI client directly with the secret key
api_key = st.secrets["GOOGLE_API_KEY"].strip()
client = genai.Client(api_key=api_key)

# 2. Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Process URLs & Build FAISS Index
if process_url_clicked:
    valid_urls = [u.strip() for u in urls if u.strip()]
    if valid_urls:
        main_placeholder.text("Loading Data... ⏳")
        data = WebBaseLoader(valid_urls).load()
        
        main_placeholder.text("Splitting Text... ✂️")
        docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(data)
        
        main_placeholder.text("Building Vector Index... ⚡")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(file_path)
        main_placeholder.text("Processing Complete! ✅")
    else:
        st.sidebar.warning("Please provide at least one valid URL.")

# 4. Question Input & RAG Pipeline
query = st.text_input("Question:", key="main_user_query_input")

if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(query)
        
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        main_placeholder.text("Calling LLM... ⏳")
        
        prompt = (
            f"You are a helpful news assistant. Use the following context to answer the question:\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Candidate models ordered by priority
        candidate_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]

        response_text = None
        used_model = None

        for model_name in candidate_models:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    if response and response.text:
                        response_text = response.text
                        used_model = model_name
                        break
                except ServerError:
                    # Exponential backoff for temporary 503 spikes
                    time.sleep(2 * (attempt + 1))
                except Exception:
                    break
            if response_text:
                break

        main_placeholder.empty()

        if response_text:
            st.header("Answer:")
            st.write(response_text)
            
            st.caption(f"Generated using `{used_model}`")
            
            st.subheader("Sources:")
            for doc in retrieved_docs:
                if "source" in doc.metadata:
                    st.write(doc.metadata["source"])
        else:
            st.error("Google's servers are currently experiencing high traffic. Please wait 10–20 seconds and retry.")
    else:
        st.warning("Please process the URLs first to build the index before asking questions.")
