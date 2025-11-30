# 📊 Enterprise Financial Analyst AI Agent

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io/)
[![Llama-3](https://img.shields.io/badge/Model-Llama--3--8B-purple.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green.svg)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/Deployed%20on-Spaces-yellow.svg)](https://huggingface.co/spaces/Shehab-Hegab/Tesla-Financial-Analyst)

A specialized Generative AI agent capable of analyzing complex 100+ page financial reports (10-Ks) with high precision. This project combines **Fine-Tuning** (for professional behavior) and **RAG** (Retrieval-Augmented Generation) for factual accuracy.

### 🔗 Quick Links
*   🚀 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Shehab-Hegab/Tesla-Financial-Analyst)**
*   🎥 **[Watch Demo Video & Post on LinkedIn](https://www.linkedin.com/posts/shehab-hegab%F0%9F%87%B5%F0%9F%87%B8-5303491b7_generativeai-llama3-rag-activity-7400637766734217216-nnIM?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJrF7QBd9Oce1lYAjrVMkebIhqFwZLmCwk)**

---

## 💡 The Problem
Generic Large Language Models (LLMs) like ChatGPT are "generalists." When asked about specific financial data (e.g., Tesla's 2023 Revenue), they often:
1.  **Hallucinate** incorrect numbers.
2.  **Fail to understand** complex corporate accounting policies (ASC standards).
3.  **Lack access** to private or recent documents.

## 🛠️ The Solution
I built a specialized **Financial Analyst Agent** that solves this using a Hybrid Architecture:

1.  **Fine-Tuning (The Brain):** Trained **Llama-3-8B** using **Unsloth (QLoRA)** on a Financial QA dataset. This taught the model to adopt a professional "Senior Analyst" persona, preventing casual or vague responses.
2.  **RAG System (The Memory):** Integrated **LangChain** and **FAISS** to chunk, index, and retrieve exact paragraphs from uploaded PDFs. The model answers *only* based on the retrieved context.

---

## 🚀 Key Features

*   **📄 Instant PDF Analysis:** Ingests massive Annual Reports (10-K) and chunks them into vector embeddings in seconds.
*   **✅ Zero Hallucinations:** Answers are strictly grounded in the provided document. If the info isn't there, the agent admits it.
*   **🔍 Source Citations:** The UI provides an expandable "Source Context" view, allowing users to verify which paragraph the AI used to generate the answer.
*   **🧠 Professional Persona:** Uses specific financial terminology (e.g., "Impairment losses," "Indefinite-lived intangible assets") thanks to fine-tuning.
*   **⚡ Optimized Performance:** Uses `InferenceClient` streaming for low-latency responses, even on CPU-only environments.

---

## 🏗️ Technical Architecture

| Component | Tool Used | Purpose |
| :--- | :--- | :--- |
| **LLM** | Meta Llama-3-8B Instruct | The reasoning engine for generating answers. |
| **Fine-Tuning** | Unsloth (QLoRA) | Optimized training to adapt the model to financial tasks 2x faster. |
| **Orchestration** | LangChain | Connects the PDF loader, Vector DB, and LLM. |
| **Vector DB** | FAISS | Stores text chunks as embeddings for semantic search. |
| **Embeddings** | Sentence-Transformers | Converts text into numerical vectors (`all-MiniLM-L6-v2`). |
| **Frontend** | Streamlit | Provides the interactive web UI. |
| **Deployment** | Hugging Face Spaces | Cloud hosting for the live application. |

---

## 📸 Screenshots

<img width="1853" height="995" alt="image" src="https://github.com/user-attachments/assets/206e0041-f1ab-457c-966c-dfdea4f19efb" />
<img width="1851" height="967" alt="image" src="https://github.com/user-attachments/assets/9844483f-e8f5-494c-bb56-7ae3b44334d3" />

---

## 💻 Installation & Usage (Local)

If you want to run this project locally on your machine:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Shehab-Hegab/Tesla-Financial-Analyst.git
    cd Tesla-Financial-Analyst
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**
    Create a `.env` file or export your Hugging Face Token (must have Write permissions):
    ```bash
    export HF_TOKEN="your_huggingface_token_here"
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Shehab-Hegab/Tesla-Financial-Analyst/issues).

## 📜 License

This project is licensed under the Apache 2.0 License.

---

**Built by [Shehab Hegab](https://github.com/Shehab-Hegab)**
