# 🩺 Medical AI Assistant

A conversational Medical AI Chatbot built with **Streamlit**, powered by **LLaMA 3.1-8B** and a **RAG knowledge base**.

## Features
- 🔍 Intent Classification — understands greetings, medical queries, and off-topic messages
- 📚 RAG Knowledge Base — 15 built-in medical topics for context retrieval
- 🧠 Conversation Memory — remembers context across multi-turn conversations
- 💡 Example questions to get started quickly

## Setup on Streamlit Cloud

1. Fork or upload this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub
3. Deploy the app and go to **Settings → Secrets**
4. Add your secret:
```
HF_API_TOKEN = "your_huggingface_token_here"
```

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This chatbot is for **informational purposes only** and is not a substitute for professional medical advice.
