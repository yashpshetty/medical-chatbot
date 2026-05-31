import os
import requests
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
# HARDCODED KNOWLEDGE BASE
# ============================================================
KNOWLEDGE_BASE = [
    "Fever is a temporary increase in body temperature, often due to an illness. A fever is usually harmless and may actually be a good sign that your body is fighting off an infection. Fever is generally defined as a temperature above 38°C (100.4°F). Common causes include viral infections, bacterial infections, heat exhaustion, and certain medications. Treatment includes rest, fluids, and fever-reducing medications like paracetamol or ibuprofen.",

    "A cough is a reflex action to clear your airways of mucus and irritants such as dust or smoke. It can be acute (short-term) or chronic (long-term). Common causes include common cold, flu, asthma, acid reflux, and smoking. A dry cough produces no phlegm; a wet/productive cough brings up mucus. Treatment depends on the cause: antihistamines for allergies, inhalers for asthma, or antibiotics for bacterial infections.",

    "A headache is pain or discomfort in the head or face area. Types include tension headaches, migraines, cluster headaches, and sinus headaches. Tension headaches are the most common and cause a dull, aching sensation. Migraines involve throbbing pain, often with nausea and sensitivity to light. Common triggers include stress, dehydration, lack of sleep, and caffeine. Treatment includes pain relievers, rest, hydration, and stress management.",

    "Diabetes mellitus is a group of diseases that result in too much sugar in the blood (high blood glucose). Type 1 diabetes means the body produces no insulin. Type 2 diabetes means the body does not use insulin properly. Symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow-healing sores. Management involves diet, exercise, blood sugar monitoring, and medications such as insulin or metformin.",

    "Hypertension (high blood pressure) is a condition where the force of blood against artery walls is consistently too high. Normal blood pressure is below 120/80 mmHg. It often has no symptoms but can lead to heart disease, stroke, and kidney failure. Risk factors include obesity, smoking, excess salt, stress, and family history. Treatment involves lifestyle changes and medications like ACE inhibitors, beta-blockers, or diuretics.",

    "The common cold is a viral infection of the upper respiratory tract. Symptoms include runny nose, sneezing, sore throat, cough, and mild fever. It is usually caused by rhinoviruses. There is no cure; treatment is supportive: rest, fluids, and over-the-counter medications for symptom relief. It typically resolves within 7-10 days. Frequent handwashing is the best prevention.",

    "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms include fever, chills, muscle aches, cough, congestion, and fatigue. It spreads through droplets when infected people cough or sneeze. Annual flu vaccination is recommended. Antiviral medications like oseltamivir (Tamiflu) can reduce severity if taken early.",

    "A sore throat is pain, scratchiness, or irritation of the throat that worsens when swallowing. It is most commonly caused by viral infections (cold, flu) or bacterial infections (strep throat). Symptoms include pain when swallowing, swollen glands, and hoarse voice. Viral sore throats resolve on their own; bacterial infections may require antibiotics. Gargling with warm salt water and staying hydrated can help.",

    "Nausea is an unpleasant sensation of unease in the stomach, often preceding vomiting. Common causes include motion sickness, pregnancy, food poisoning, and medications. Treatment includes rest, clear fluids, bland foods (banana, rice, applesauce, toast), and antiemetic medications. Severe or persistent nausea with vomiting should be evaluated by a doctor to prevent dehydration.",

    "Diarrhea is loose, watery stools occurring more than three times a day. Common causes include viral infections, bacterial infections, food intolerance, and irritable bowel syndrome. Key concern is dehydration. Oral rehydration solutions (ORS) are the primary treatment. Seek medical attention if diarrhea is severe, contains blood, or lasts more than two days.",

    "Asthma is a condition in which airways narrow and swell, making breathing difficult. Symptoms include shortness of breath, wheezing, coughing, and chest tightness. Triggers include allergens, cold air, exercise, and smoke. It is managed with bronchodilator inhalers for acute symptoms and corticosteroid inhalers for long-term prevention. Severe asthma attacks require emergency care.",

    "Allergies occur when the immune system reacts to a foreign substance such as pollen, pet dander, or food. Symptoms include sneezing, runny nose, itchy eyes, hives, and in severe cases, anaphylaxis. Treatment includes antihistamines, decongestants, corticosteroids, and avoiding known allergens. Severe allergies may require an epinephrine auto-injector (EpiPen).",

    "Back pain is one of the most common medical complaints. It can result from muscle strain, bulging discs, arthritis, or osteoporosis. Symptoms range from a dull ache to a stabbing sensation. Risk factors include age, lack of exercise, excess weight, and poor posture. Most back pain improves with rest, hot/cold packs, and over-the-counter pain relievers. Severe or persistent pain needs medical evaluation.",

    "Dehydration occurs when you lose more fluids than you take in. Symptoms include extreme thirst, dark urine, dizziness, dry mouth, and fatigue. Causes include excessive sweating, vomiting, diarrhea, and insufficient fluid intake. Treatment is rehydration with water or oral rehydration solutions. Severe dehydration requires IV fluids and immediate medical attention.",

    "Anxiety is a feeling of fear, dread, and uneasiness that may cause sweating, restlessness, and rapid heartbeat. Anxiety disorders involve excessive and persistent worry. Types include generalized anxiety disorder, panic disorder, and social anxiety disorder. Treatment includes cognitive behavioral therapy (CBT), medications (SSRIs), mindfulness, and regular exercise."
]

# ============================================================
# VECTORIZER — cached so it only loads once
# ============================================================
@st.cache_resource
def load_vectorizer():
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(KNOWLEDGE_BASE)
    return vectorizer, doc_vectors

vectorizer, doc_vectors = load_vectorizer()

# ============================================================
# CORE FUNCTIONS
# ============================================================
def query_llm(messages, max_tokens=400):
    if not HF_API_TOKEN:
        return "⚠️ HF_API_TOKEN not set. Go to Settings → Secrets and add your token."
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            return f"⚠️ API Error: {result['error']}"
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def retrieve_context(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)
    return KNOWLEDGE_BASE[scores.argmax()]

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
    result = query_llm(messages, max_tokens=10).strip().lower()
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
        return "👋 Hello! I am your Medical AI Assistant. How can I help you today? Please describe your symptoms or ask a health-related question."
    elif intent == "non_medical":
        return "🩺 I am a Medical AI Assistant. Please ask health-related questions only."
    return rag_query(user_input, chat_history)

# ============================================================
# SESSION STATE — stores chat between reruns
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # internal LLM memory
if "messages" not in st.session_state:
    st.session_state.messages = []       # display messages

# ============================================================
# UI
# ============================================================
st.title("🩺 Medical AI Assistant")
st.caption("Powered by LLaMA 3.1 + RAG Knowledge Base")
st.warning("⚠️ This chatbot is for informational purposes only and is not a substitute for professional medical advice.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
        st.markdown(msg["content"])

# Example questions (shown only when chat is empty)
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

# Handle example button click
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

# Chat input box (always at the bottom)
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

# Clear button in sidebar
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This assistant uses LLaMA 3.1-8B and a local medical knowledge base to answer health questions.")
    st.markdown("**Topics covered:**")
    st.markdown("Fever, Cough, Headache, Diabetes, Hypertension, Flu, Sore throat, Nausea, Diarrhea, Asthma, Allergies, Back pain, Dehydration, Anxiety")
