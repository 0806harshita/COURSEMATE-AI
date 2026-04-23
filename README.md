# 🚀 CourseMate AI

An AI-powered document assistant that helps users interact with their files (PDF, TXT, etc.) using natural language queries. Built using modern LLM and vector database techniques for efficient information retrieval.

---

## 📌 Features

* 📄 Upload and process documents (PDF, TXT)
* 🤖 Ask questions from your documents using AI
* 🔍 Semantic search using vector embeddings
* ⚡ Fast retrieval with vector database
* 🧠 Context-aware responses (RAG - Retrieval Augmented Generation)

---

## 🛠️ Tech Stack

* **Backend:** Python
* **LLM Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Mistral AI
* **Frontend :** Streamlit

---

## 📂 Project Structure

```
CourseMate AI/
│── .vscode/                # VS Code settings
│── chroma_db/             # Vector database storage
│── document loaders/      # Load PDF, TXT files
│── retrievers/            # Retrieval logic
│── vector_store/          # Embedding & storage logic
│── .env                   # API keys (DO NOT SHARE)
│── app.py                 # Main app entry point
│── create_db.py           # Script to create vector DB
│── main2_pdf.py           # PDF handling module
│── main_txt.py            # TXT handling module
│── main_file_important.py # Core logic
│── requirements.txt       # Dependencies
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/0806harshita/COURSEMATE-AI.git
cd COURSEMATE-AI
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add your API key:

```
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Usage

### Step 1: Create Vector Database

```bash
python create_db.py
```

### Step 2: Run the Application

```bash
python app.py
```

---

## 🧠 How It Works

1. Documents are loaded using custom loaders
2. Text is converted into embeddings
3. Stored in ChromaDB vector database
4. User query is embedded and matched
5. Relevant context is retrieved
6. LLM generates accurate answer

---

## 📌 Future Improvements

* 🌐 Web UI with React
* 📊 Support for more file formats (DOCX, PPT)
* 🔐 Authentication system
* ☁️ Cloud deployment

---

## 🤝 Contributing

Feel free to fork this repository and contribute!

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ Acknowledgements

* LangChain
* OpenAI
* ChromaDB

---

## 👩‍💻 Author

**Harshita Kumari**
GitHub: https://github.com/0806harshita

---
