import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # Updated Import
from huggingface_hub import InferenceClient
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tesla Financial Analyst", layout="wide")
st.title("📊 Tesla Financial Analyst AI")
st.markdown("This Agent uses **Llama-3** to analyze the Tesla 10-K Report.")

# --- SIDEBAR: SETTINGS ---
st.sidebar.header("Configuration")
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Please set your HF_TOKEN in the Space Settings!")
    st.stop()

# Advanced settings
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 100, 50)
num_chunks = st.sidebar.slider("Number of Retrieved Chunks", 3, 15, 6, 1)

# --- 1. SETUP THE MODEL ---
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
client = InferenceClient(model=repo_id, token=hf_token)

# --- HELPER FUNCTIONS ---

def preprocess_query(query):
    """Enhance query for better retrieval"""
    financial_terms = {
        "revenue": "revenue sales income",
        "profit": "profit earnings net income",
        "debt": "debt liabilities obligations",
        "cash": "cash liquidity cash flow",
        "assets": "assets resources holdings",
        "growth": "growth increase expansion",
        "risk": "risk uncertainty threat challenge"
    }
    
    query_lower = query.lower()
    for term, expansion in financial_terms.items():
        if term in query_lower:
            query = f"{query} {expansion}"
            break
    return query

def rerank_documents(query, docs, top_k=None):
    """Simple reranking based on keyword overlap"""
    if top_k is None:
        top_k = len(docs)
    
    query_words = set(re.findall(r'\w+', query.lower()))
    
    scored_docs = []
    for doc in docs:
        content_words = set(re.findall(r'\w+', doc.page_content.lower()))
        overlap = len(query_words.intersection(content_words))
        length_score = min(len(doc.page_content) / 1000, 1.0)
        score = overlap * 10 + length_score
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_docs[:top_k]]

def build_context(docs, max_tokens=3000):
    context_parts = []
    current_length = 0
    for i, doc in enumerate(docs):
        doc_tokens = len(doc.page_content) // 4
        if current_length + doc_tokens > max_tokens:
            break
        context_parts.append(f"[Source {i+1}]:\n{doc.page_content}")
        current_length += doc_tokens
    return "\n\n".join(context_parts)

# --- 2. FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload Tesla PDF", type="pdf")

if "db" not in st.session_state:
    st.session_state.db = None

if uploaded_file and st.session_state.db is None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Indexing Document..."):
        try:
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            docs = text_splitter.split_documents(pages)
            
            # Updated Embedding Class
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            st.session_state.db = FAISS.from_documents(docs, embeddings)
            st.sidebar.success(f"✅ Ready! Indexed {len(docs)} chunks.")
            
        except Exception as e:
            st.sidebar.error(f"Error indexing: {e}")

# --- 3. CHAT INTERFACE ---
user_query = st.text_input("Ask a question:", placeholder="What was the total revenue in 2023?")

if user_query and st.session_state.db:
    with st.spinner("Thinking..."):
        enhanced_query = preprocess_query(user_query)
        relevant_docs = st.session_state.db.similarity_search(enhanced_query, k=num_chunks * 2)
        reranked_docs = rerank_documents(user_query, relevant_docs, top_k=num_chunks)
        context = build_context(reranked_docs)
        
        system_prompt = """You are a senior financial analyst. Answer the question strictly based on the context.
If the answer is not in the context, say "I cannot find that information in the document." """
        
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}"
        
        st.write("### 🤖 Analyst Response:")
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            with st.expander("View Source Context"):
                st.write(context)

        except Exception as e:
            st.error(f"API Error: {e}")

elif user_query and not st.session_state.db:
    st.warning("Please upload a PDF first.")
