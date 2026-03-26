# 🏏 CricBot AI – Intelligent Cricket Chatbot

CricBot is an AI-powered cricket assistant that provides accurate and real-time cricket information using modern AI technologies like Retrieval-Augmented Generation (RAG), Vector Search, and LLM-based reranking.

---

## 🚀 Features

* 🔍 Semantic Search using Pinecone Vector Database
* 🧠 LLM-based Reranking for improved accuracy
* 🤖 Intelligent responses powered by Groq (LLM)
* 📊 Player Information Retrieval
* 📅 Match Schedule Lookup (Local JSON)
* 🌐 Live Cricket Data (API Integration Ready)
* 💬 Interactive Chat UI using Streamlit

---

## 🧠 Tech Stack

* Python
* Pinecone (Vector Database)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* Groq LLM (`llama-3.3-70b-versatile`)

---

## 🏗️ Architecture

User Query
→ Embedding (HuggingFace)
→ Pinecone Vector Search (Top K Results)
→ 🧠 Reranking using LLM
→ Context Selection
→ 🤖 Final Answer Generation (Groq LLM)

---

## 📂 Project Structure

```bash
chatbot/
│── search.py               # Main chatbot logic (RAG + reranking)
│── upload_to_pinecone.py   # Data indexing script
│── live_cricket.py         # Live API / mock functions
│── players.json            # Player dataset
│── international.json      # International matches
│── league.json             # League/IPL matches
│── women.json              # Women's matches
│── .env                    # API keys (ignored)
│── .gitignore              # Ignore sensitive files
│── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/cricbot-ai.git
cd cricbot-ai
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Add Environment Variables

Create a `.env` file and add:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

---

### 4️⃣ Upload Data to Pinecone

```bash
python upload_to_pinecone.py
```

---
```

---

## 🧪 Example Queries

* Who is Virat Kohli
* Best captain in India
* Live cricket score
* India next match
* Latest cricket news

---

## 📈 Improvements

* Added LLM-based reranking to improve relevance
* Reduced noise from vector search results
* Improved chatbot response quality

---

## 🔐 Security

* API keys stored securely in `.env`
* `.gitignore` prevents sensitive data exposure

---

## 🌟 Future Enhancements

* Real-time Cricbuzz API integration
* Chat memory (conversation context)
* Voice-based interaction
* Deployment (public web app)

---

## 👩‍💻 Author

**Akshayaa**
---

## 💖 Acknowledgement

Built as part of an AI chatbot internship project to demonstrate real-world RAG architecture with reranking.

---
