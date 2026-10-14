import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("News Research Tool 📈")

st.sidebar.title("News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store_index"
main_placeholder = st.empty()

# Safely set the environment variable from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"].strip()

# Initialize LLM with standard model name
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"].strip(),
    temperature=0.2
)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process URLs
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

# Query section
query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        
        system_prompt = (
            "You are a helpful news assistant. "
            "Answer the question strictly using the provided context.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        main_placeholder.text("Calling LLM... ✅")
        response = rag_chain.invoke({"input": query})
        
        st.header("Answer:")
        st.write(response["answer"])
