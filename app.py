import os
import streamlit as st
from google import genai
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_index"
main_placeholder = st.empty()

# Initialize Google GenAI client directly with the secret key
api_key = st.secrets["GOOGLE_API_KEY"].strip()
client = genai.Client(api_key=api_key)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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

query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        retrieved_docs = retriever.invoke(query)
        
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        main_placeholder.text("Calling LLM... ✅")
        
        prompt = (
            f"You are a helpful news research assistant. Use the following context to answer the question:\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
       # Calls the official endpoint with gemini-3.8-flash
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        
        st.header("Answer:")
        st.write(response.text)
        
        st.subheader("Sources:")
        for doc in retrieved_docs:
            if "source" in doc.metadata:
                st.write(doc.metadata["source"])
    else:
        st.warning("Please process the URLs first to build the index before asking questions.")
