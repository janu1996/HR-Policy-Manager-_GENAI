# HR Policy Assistant – GenAI Chatbot

An AI-powered chatbot that helps employees find answers to HR policy questions using Retrieval-Augmented Generation (RAG).

## 📌 Project Overview

The HR Policy Assistant uses uploaded HR policy documents to provide relevant, context-based answers. It helps employees access company policies quickly without manually searching through lengthy PDF files.

## ✨ Features

- AI-powered HR policy question answering
- RAG-based document retrieval
- Supports eight HR policy PDF documents
- Source document and page references
- Fallback responses for questions not covered by the policies
- Basic sensitive-query detection and HR escalation guidance
- Interactive Streamlit chat interface
- Knowledge base rebuilding option
- Conversation history and suggested questions

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- Groq API
- Retrieval-Augmented Generation (RAG)

## 📚 HR Policy Documents

The knowledge base covers:

1. Leave Policy
2. Reimbursement Policy
3. Travel Policy
4. Code of Conduct
5. Work From Home Policy
6. Attendance Policy
7. Employee Grievance Policy
8. Sensitive Matters and HR Escalation Policy

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd HR-Policy-Assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Groq API key and the model configuration required by the application.

**Never upload your `.env` file or expose your API key publicly.**

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🎯 Objective

To provide employees with a convenient AI assistant that simplifies HR policy access and generates answers grounded in the available policy documents.

## 🔒 Note

Responses depend on the uploaded HR policy documents. Sensitive matters may require direct assistance from the HR team.

## 👩‍💻 Project Status

Locally tested and uploaded to GitHub. Deployment is planned for a later stage.
