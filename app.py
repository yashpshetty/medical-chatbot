import os
import requests
import wikipedia
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# CONFIGURATION
# ============================================================
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ============================================================
# WIKIPEDIA KNOWLEDGE BASE — loaded once and cached
# ============================================================

# Fallback hardcoded entries in case any Wikipedia topic fails
FALLBACK = {
    "Fever": "Fever is a temporary increase in body temperature above 38C, usually caused by infections. Treatment includes rest, fluids, and paracetamol or ibuprofen.",
    "Diabetes mellitus": "Diabetes is a condition of high blood glucose. Type 1 lacks insulin; Type 2 resists it. Managed with diet, exercise, and medications like metformin.",
    "Common cold": "The common cold is a viral upper respiratory infection causing runny nose, sore throat, and cough. It resolves in 7-10 days with rest and fluids.",
    "Allergic disease": "Allergies are immune reactions to substances like pollen or food. Symptoms include sneezing, hives, and itchy eyes. Treated with antihistamines.",
}

TOPICS = [
    "Fever", "Cough", "Headache", "Diabetes mellitus",
    "Hypertension", "Common cold", "Influenza", "Sore throat",
    "Nausea", "Diarrhea", "Asthma", "Allergic disease",
    "Back pain", "Dehydration", "Anxiety"
]

@st.cache_resource(show_spinner="📚 Loading medical knowledge from Wikipedia...")
def load_knowledge_base():
    documents = []
    loaded = []
    failed = []
    for topic in TOPICS:
        try:
            text = wikipedia.summary(topic, sentences=6, auto_suggest=False)
            documents.append(text)
            loaded.append(topic)
        except Exception:
            # Use fallback if available, else skip
            if topic in FALLBACK:
                documents.append(FALLBACK[topic])
                loaded.append(f"{topic} (fallback)")
            else:
                failed.append(topic)

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    return vectorizer, doc_vectors, documents, loaded, failed

vectorizer, doc_vectors, wiki_documents, loaded_topics, failed_topics = load_knowledge_base()

# ============================================================
# CORE FUNCTIONS
# ============================================================
def query_llm(messages, max_tokens=500):
    if not HF_API_TOKEN:
        return "HF_API_TOKEN not set. Go to Settings -> Secrets and add your token."
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            return f"API Error: {result['error']}"
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def retrieve_context(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)
    return wiki_documents[scores.argmax()]

def classify_intent(user_input, chat_history):
    history_text = ""
    if chat_history:
        history_text = "\n\nRecent conversation:\n"
        for turn in chat_history[-4:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an intent classifier. Classify the user message into ONE of:\n"
                "- greeting\n- medical\n- non_medical\n\n"
                "If the assistant previously asked a follow-up medical question, classify the reply as 'medical'.\n"
                f"{history_text}\n"
                "Reply with ONLY one word: greeting, medical, or non_medical."
            )
        },
        {"role": "user", "content": f"Classify: {user_input}"}
    ]
    result = query_llm(messages, max_tokens=50).strip().lower()
    if "greeting" in result:
        return "greeting"
    elif "non_medical" in result or "non-medical" in result:
        return "non_medical"
    return "medical"

def rag_query(user_input, chat_history):
    context = retrieve_context(user_input)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Medical AI Assistant. Use the context to answer.\n"
                "Be concise, empathetic, and medically responsible.\n"
                "If something sounds serious, advise the user to consult a doctor.\n\n"
                f"Medical Context:\n{context}"
            )
        }
    ]
    for turn in chat_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": user_input})
    return query_llm(messages, max_tokens=400)

def medical_bot(user_input, chat_history):
    intent = classify_intent(user_input, chat_history)
    if intent == "greeting":
        return "Hello! I am your Medical AI Assistant. How can I help you today? Please describe your symptoms or ask a health-related question."
    elif intent == "non_medical":
        return "I am a Medical AI Assistant. Please ask health-related questions only."
    return rag_query(user_input, chat_history)

# ============================================================
# SESSION STATE
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # current active LLM memory
if "messages" not in st.session_state:
    st.session_state.messages = []           # current active display messages
if "saved_sessions" not in st.session_state:
    st.session_state.saved_sessions = []     # list of past saved sessions
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 1     # session numbering

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🕒 Chat History")

    # NEW CHAT — saves current chat to history, starts fresh
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        if st.session_state.chat_history:
            # Save current session before clearing
            st.session_state.saved_sessions.append({
                "id": st.session_state.session_counter,
                "title": st.session_state.chat_history[0]["user"][:35] + "..."
                         if len(st.session_state.chat_history[0]["user"]) > 35
                         else st.session_state.chat_history[0]["user"],
                "chat_history": st.session_state.chat_history.copy(),
                "messages": st.session_state.messages.copy()
            })
            st.session_state.session_counter += 1
        # Clear current chat for fresh start
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # CURRENT CHAT section
    if st.session_state.chat_history:
        st.markdown("**Current Chat**")

        # Clear Text — clears only display, keeps LLM memory intact
        if st.button("🧹 Clear Text", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")

    # PAST SESSIONS section
    if st.session_state.saved_sessions:
        st.markdown("**Past Sessions**")

        if st.button("🗑️ Delete All History", use_container_width=True):
            st.session_state.saved_sessions = []
            st.session_state.session_counter = 1
            st.rerun()

        st.markdown("")

        for i, session in enumerate(reversed(st.session_state.saved_sessions)):
            real_index = len(st.session_state.saved_sessions) - 1 - i
            with st.container(border=True):
                st.markdown(f"**Chat {session['id']}: {session['title']}**")
                st.caption(f"{len(session['chat_history'])} messages")
                col1, col2 = st.columns(2)
                # Restore session
                if col1.button("↩️ Restore", key=f"restore_{i}", use_container_width=True):
                    st.session_state.chat_history = session["chat_history"].copy()
                    st.session_state.messages = session["messages"].copy()
                    st.session_state.saved_sessions.pop(real_index)
                    st.rerun()
                # Delete session
                if col2.button("❌ Delete", key=f"del_{i}", use_container_width=True):
                    st.session_state.saved_sessions.pop(real_index)
                    st.rerun()
    else:
        if not st.session_state.chat_history:
            st.info("No history yet.\nStart a conversation!")

# ============================================================
# MAIN UI
# ============================================================
st.title("🩺 Medical AI Assistant")
st.caption("Powered by LLaMA 3.1 + RAG Knowledge Base")
st.warning("This chatbot is for informational purposes only and is not a substitute for professional medical advice.")

# Display current messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
        st.markdown(msg["content"])

# Example questions when chat is empty
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    examples = [
        "I have a fever and headache since yesterday",
        "What are the symptoms of diabetes?",
        "I have a dry cough and sore throat",
        "How to manage high blood pressure?",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state.pending_input = example
            st.rerun()

# Handle example button clicks
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Thinking..."):
            response = medical_bot(user_input, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append({"user": user_input, "bot": response})
    st.rerun()

# Chat input
user_input = st.chat_input("Describe your symptoms or ask a health question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Thinking..."):
            response = medical_bot(user_input, st.session_state.chat_history)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append({"user": user_input, "bot": response})
