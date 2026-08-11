# News Research Tool 📈 (RAG Assistant)

An end-to-end Retrieval-Augmented Generation (RAG) web application built to analyze news articles and answer complex research queries grounded in source context.

## 🚀 Features
* **Dynamic Article Scraping:** Extract and parse live news content directly from web URLs.
* **Vector Indexing:** Chunk unstructured text and generate high-dimensional embeddings stored in a local FAISS vector database.
* **Semantic Context Retrieval:** Fetch top relevant passages via cosine similarity to reduce LLM hallucinations.
* **Grounded Answer Generation:** Leverage Google Gemini LLM to answer user queries with explicit source citations.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM:** Google Gemini (`langchain-google-genai`)
* **Framework:** LangChain & LangChain Classic
* **Vector Database:** FAISS (`faiss-cpu`)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Frontend / Deployment:** Streamlit & Streamlit Community Cloud

## 🔧 Installation & Local Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/VaishnaviB45/news-research-tool.git](https://github.com/VaishnaviB45/news-research-tool.git)
   cd news-research-tool
