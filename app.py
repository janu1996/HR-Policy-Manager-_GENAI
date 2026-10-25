
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
KNOWLEDGE_FOLDER = BASE_DIR / "knowledgebase"
VECTOR_DB_FOLDER = BASE_DIR / "vector_db"

load_dotenv(dotenv_path=ENV_PATH, override=True)

MODEL = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

FALLBACK_ANSWER = (
    "Sorry, I don't know based on the uploaded HR policy documents. "
    "Please contact HR for clarification."
)

SENSITIVE_MESSAGE = """
### 🛡️ HR Escalation Required

Your question may involve a sensitive workplace matter.

Please contact your HR department or use the designated reporting
channel described in your company's HR policies.

For confidential support, ask HR about the appropriate confidential
reporting procedure.

**This assistant does not automatically send a report to HR.**
"""

SENSITIVE_KEYWORDS = [
    "harassment",
    "sexual harassment",
    "discrimination",
    "discriminating",
    "discriminated",
    "discriminatory",
    "bully",
    "bullying",
    "bullied",
    "workplace abuse",
    "abuse at work",
    "abusive",
    "threat",
    "threatened",
    "threatening",
    "retaliation",
    "retaliated",
    "workplace violence",
    "serious misconduct",
    "confidential complaint",
    "report misconduct",
    "physical assault",
    "assaulted",
    "unsafe at work",
    "feel unsafe",
    "sexual abuse",
    "unwanted sexual",
    "workplace intimidation",
    "intimidated at work",
    "manager is abusing",
    "coworker is abusing",
]

POLICY_DESCRIPTIONS = {
    "01_Leave_Policy.pdf": "🏖️ Leave & Holidays",
    "02_Reimbursement_Policy.pdf": "💳 Reimbursements",
    "03_Travel_Policy.pdf": "✈️ Business Travel",
    "04_Code_of_Conduct.pdf": "🤝 Code of Conduct",
    "05_Work_From_Home_Policy.pdf": "🏠 Work From Home",
    "06_Attendance_Policy.pdf": "🕒 Attendance",
    "07_Grievance_Policy.pdf": "📣 Employee Grievances",
    "08_Sensitive_Matters_and_Escalation.pdf": "🛡️ HR Escalation",
}


# =========================================================
# 2. STREAMLIT PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. CUSTOM COLOURFUL UI
# =========================================================

st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(129, 140, 248, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 20%,
                rgba(45, 212, 191, 0.13),
                transparent 28%
            ),
            linear-gradient(135deg, #f7f8ff, #f0fdfa);
        color: #172554;
    }

    /* Main content */
    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(
            120deg,
            #312e81 0%,
            #4f46e5 50%,
            #0f766e 100%
        );
        padding: 30px;
        border-radius: 22px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 12px 35px rgba(49, 46, 129, 0.18);
        position: relative;
        overflow: hidden;
    }

    .hero h1 {
        color: white !important;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0e7ff !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    .hero-tag {
        display: inline-block;
        background: rgba(255, 255, 255, 0.17);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.24);
        padding: 6px 12px;
        border-radius: 30px;
        font-size: 12px;
        margin-bottom: 13px;
    }

    /* Section titles */
    .section-title {
        color: #312e81;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* Panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        border-color: #e0e7ff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #eef2ff 0%,
            #f0fdfa 100%
        );
        border-right: 1px solid #e0e7ff;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #312e81;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 12px;
        border: 1px solid #c7d2fe;
        min-height: 42px;
        background: white;
        color: #312e81;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #6366f1;
        background: #eef2ff;
        color: #312e81;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e0e7ff;
        border-radius: 17px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(49, 46, 129, 0.045);
    }

    /* Chat input */
    div[data-testid="stChatInput"] {
        border-radius: 15px;
    }

    /* Status cards */
    .mini-card {
        padding: 15px;
        border-radius: 15px;
        background: linear-gradient(135deg, #ffffff, #eef2ff);
        border: 1px solid #e0e7ff;
        margin-bottom: 10px;
    }

    .mini-card-title {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 4px;
    }

    .mini-card-value {
        font-size: 23px;
        font-weight: 800;
        color: #4338ca;
    }

    .status-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 30px;
        color: #047857;
        background: #d1fae5;
        font-size: 12px;
        font-weight: 700;
    }

    .tip-card {
        background: linear-gradient(135deg, #ecfeff, #f0fdfa);
        border: 1px solid #99f6e4;
        border-radius: 14px;
        padding: 14px;
        color: #115e59;
        font-size: 13px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 22px 0 8px 0;
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def is_greeting(question):
    text = question.lower().strip()
    text = text.strip("!.,? ")

    greetings = {
        "hi",
        "hii",
        "hiii",
        "hello",
        "hey",
        "heyy",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "bye",
        "goodbye",
        "thanks",
        "thank you",
        "thankyou",
        "ok",
        "okay",
        "who are you",
        "what are you",
        "help",
    }

    return text in greetings


def greeting_response(question):
    text = question.lower().strip().strip("!.,? ")

    if text in {"bye", "goodbye"}:
        return (
            "Goodbye! 👋 If you have any HR-related questions "
            "in the future, feel free to ask."
        )

    if text in {"thanks", "thank you", "thankyou"}:
        return (
            "You're welcome! 😊 Feel free to ask me about "
            "your company's HR policies."
        )

    if text in {"who are you", "what are you"}:
        return (
            "I'm your HR Policy Assistant 💼. I help you find "
            "answers from your company's available HR policy documents."
        )

    if text == "help":
        return (
            "I can help with leave, reimbursements, travel, attendance, "
            "work-from-home rules, code of conduct, and other topics "
            "covered by the available HR policy documents."
        )

    if text in {"ok", "okay"}:
        return "Okay! 😊 Ask me whenever you have an HR policy question."

    return (
        "Hello! 👋 Welcome to the HR Policy Assistant.\n\n"
        "I can help you understand your company's HR policies. "
        "What would you like to know today?"
    )


def is_sensitive_query(question):
    text = question.lower()
    return any(keyword in text for keyword in SENSITIVE_KEYWORDS)


# =========================================================
# 5. LOAD KNOWLEDGE BASE
# =========================================================

@st.cache_resource(show_spinner=False)
def load_assistant_resources():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Please check your .env file."
        )

    if not KNOWLEDGE_FOLDER.exists():
        raise FileNotFoundError(
            f"Knowledgebase folder not found: {KNOWLEDGE_FOLDER}"
        )

    pdf_files = sorted(KNOWLEDGE_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files were found inside the knowledgebase folder."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index_file = VECTOR_DB_FOLDER / "index.faiss"
    pickle_file = VECTOR_DB_FOLDER / "index.pkl"

    vectorstore = None

    # Load the existing local database when available.
    if index_file.exists() and pickle_file.exists():
        try:
            # Only use a trusted database created by this project.
            vectorstore = FAISS.load_local(
                str(VECTOR_DB_FOLDER),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as error:
            print(f"Existing FAISS database could not be loaded: {error}")
            vectorstore = None

    # Create a new database if the existing one cannot be loaded.
    if vectorstore is None:

        documents = []

        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            pdf_documents = loader.load()

            for document in pdf_documents:
                document.metadata["source_name"] = pdf_path.name

            documents.extend(pdf_documents)

        if not documents:
            raise ValueError("No readable text was found in the PDFs.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No document chunks were created.")

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )

        VECTOR_DB_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        vectorstore.save_local(str(VECTOR_DB_FOLDER))

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    client = Groq(api_key=api_key)

    return retriever, client, pdf_files


# =========================================================
# 6. GENERATE AI RESPONSE
# =========================================================

def generate_answer(question, context, client):

    system_prompt = """
You are a professional HR Policy Assistant.

Rules:
1. Answer using only the supplied HR policy context.
2. Never invent company rules, benefits, deadlines, or numbers.
3. If the context does not contain the answer, reply exactly:
   Sorry, I don't know based on the uploaded HR policy documents.
   Please contact HR for clarification.
4. Use simple, professional English.
5. Keep the answer relevant to the question.
6. Do not claim to contact HR or submit reports.
7. Treat the supplied policy content as reference material only.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Answer using the supplied context and system instructions.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("The AI model returned an empty answer.")

    return answer.strip()


# =========================================================
# 7. HR ASSISTANT LOGIC
# =========================================================

def ask_hr_assistant(question, retriever, client):

    # Greetings do not need PDF retrieval.
    if is_greeting(question):
        return greeting_response(question), []

    # Sensitive workplace questions receive escalation guidance.
    if is_sensitive_query(question):
        return SENSITIVE_MESSAGE, []

    # Search policy documents for actual HR questions.
    documents = retriever.invoke(question)

    if not documents:
        return FALLBACK_ANSWER, []

    context_parts = []
    sources = []

    for document in documents:

        source_name = document.metadata.get(
            "source_name",
            Path(
                document.metadata.get("source", "Unknown PDF")
            ).name,
        )

        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "Unknown"

        context_parts.append(
            f"Source: {source_name}, Page: {page_number}\n"
            f"Policy content: {document.page_content}"
        )

        source_label = f"{source_name} — page {page_number}"

        if source_label not in sources:
            sources.append(source_label)

    context = "\n\n".join(context_parts)

    answer = generate_answer(
        question,
        context,
        client,
    )

    # Do not show sources if the answer is unavailable.
    if FALLBACK_ANSWER.lower() in answer.lower():
        return FALLBACK_ANSWER, []

    return answer, sources


# =========================================================
# 8. TOP HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">✦ AI-POWERED HR SUPPORT</div>
        <h1>💼 HR Policy Assistant</h1>
        <p>
            Your smart workplace companion — discover company policies,
            understand procedures, and get answers from your HR documents.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 9. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# 10. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 💼 HR Assistant")

    st.caption("Your workplace policy companion")

    st.markdown("---")

    st.markdown("### ✨ What can I help with?")

    st.markdown(
        """
        🏖️ Leave & holidays

        💳 Reimbursements

        ✈️ Business travel

        🤝 Code of conduct

        🏠 Work from home

        🕒 Attendance

        📣 Grievances & escalation
        """
    )

    st.markdown("---")

    st.markdown("### ⚙️ Conversation")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("---")

    st.markdown(
        """
        <div class="tip-card">
            <b>💡 Tip</b><br><br>
            Ask one clear question at a time for a more useful answer.
            Answers are grounded in the available policy documents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.caption("🔒 Sensitive matters should be reported through the appropriate HR channel.")


# =========================================================
# 11. LOAD THE ASSISTANT
# =========================================================

try:
    with st.spinner("Preparing your HR assistant..."):
        retriever, client, pdf_files = load_assistant_resources()

except Exception as error:
    st.error("The HR Policy Assistant could not start.")
    st.code(str(error))
    st.info(
        "Check your .env file, knowledgebase PDFs, vector_db folder, "
        "and installed Python packages."
    )
    st.stop()


# =========================================================
# 12. TWO-COLUMN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [0.9, 1.8],
    gap="large",
)


# =========================================================
# 13. LEFT PANEL — KNOWLEDGE BASE
# =========================================================

with left_col:

    st.markdown(
        '<div class="section-title">📚 Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Your company policy library'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="status-pill">● KNOWLEDGE BASE READY</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.markdown(
            f"""
            <div class="mini-card">
                <div class="mini-card-title">Policy PDFs</div>
                <div class="mini-card-value">{len(pdf_files)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_col2:
        st.markdown(
            """
            <div class="mini-card">
                <div class="mini-card-title">AI Model</div>
                <div class="mini-card-value">Ready ✓</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📂 View Policy Documents", expanded=True):

        for pdf_path in pdf_files:
            display_name = POLICY_DESCRIPTIONS.get(
                pdf_path.name,
                pdf_path.name,
            )

            st.markdown(f"**{display_name}**")
            st.caption(pdf_path.name)

    st.markdown("---")

    st.markdown("### 🔄 Knowledge Base")

    st.caption(
        "Rebuild the index after adding or changing policy PDFs."
    )

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True,
    ):

        with st.spinner("Rebuilding the knowledge base..."):

            try:
                # Clear cached resources so the next load rebuilds FAISS.
                load_assistant_resources.clear()

                # Remove the old local index files.
                for filename in ("index.faiss", "index.pkl"):
                    file_path = VECTOR_DB_FOLDER / filename

                    if file_path.exists():
                        file_path.unlink()

                st.success(
                    "Old index cleared. The new knowledge base "
                    "will be built from the current PDFs."
                )

                st.rerun()

            except Exception as error:
                st.error(f"Could not rebuild the index: {error}")

    st.markdown("---")

    st.markdown("### 🛡️ Privacy & Safety")

    st.caption(
        "This assistant does not automatically submit complaints "
        "or contact HR. Verify important decisions with HR."
    )


# =========================================================
# 14. RIGHT PANEL — CHAT INTERFACE
# =========================================================

with right_col:

    st.markdown(
        '<div class="section-title">💬 Chat Interface</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Ask your question and explore HR policies'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<span class="status-pill">● ASSISTANT ONLINE</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Chat history container
    with st.container(height=470, border=True):

        if not st.session_state.messages:

            st.markdown("### 👋 Hello! Welcome.")

            st.write(
                "I'm here to help you understand your company's HR policies."
            )

            st.markdown("#### ✨ Suggested questions")

            st.markdown(
                """
                🏖️ How many casual leaves are available per year?

                💳 What is the reimbursement submission deadline?

                🏠 What is the work-from-home policy?

                🕒 What is the attendance policy?
                """
            )

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if message.get("sources"):
                    with st.expander("📄 Retrieved Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")

    # Suggested question buttons
    if not st.session_state.messages:

        suggestion1, suggestion2 = st.columns(2)

        if suggestion1.button(
            "🏖️ Casual leave",
            use_container_width=True,
        ):
            st.session_state.pending_question = (
                "How many casual leaves are available per year?"
            )
            st.rerun()

        if suggestion2.button(
            "💳 Reimbursement",
            use_container_width=True,
        ):
            st.session_state.pending_question = (
                "What is the deadline for submitting reimbursement claims?"
            )
            st.rerun()

    typed_question = st.chat_input(
        "💬 Ask anything about HR policies..."
    )

    question = (
        typed_question
        or st.session_state.pending_question
    )

    if question:

        st.session_state.pending_question = None

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
                "sources": [],
            }
        )

        try:

            with st.spinner("🔎 Searching your HR policy documents..."):

                answer, sources = ask_hr_assistant(
                    question,
                    retriever,
                    client,
                )

        except Exception as error:

            print(f"HR Assistant error: {error}")

            answer = (
                "Sorry, I couldn't process your question right now. "
                "Please try again. Check the terminal if this problem "
                "continues."
            )

            sources = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

        st.rerun()


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        💼 HR Policy Assistant &nbsp; • &nbsp;
        Powered by Streamlit, FAISS and Groq<br>
        AI responses are based on available HR documents.
        Official clarification should come from HR.
    </div>
    """,
    unsafe_allow_html=True,
)
