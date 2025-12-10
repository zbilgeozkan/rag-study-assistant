# 📘 RAG Study Assistant — FAISS + Gemini + FastAPI + Cloud Run
**A document-based personal study assistant for students.**

This project allows a student to upload their **course lecture PDFs**, embed them using **FAISS**, and ask questions that are answered using **Google Gemini**, fully grounded in the course materials.

The system is ideal for courses where students must study using lecture slides, notes, or textbook PDFs — such as **Cloud Computing, Distributed Systems, Machine Learning**, etc.

![Rag Assistant](/assets/ragassistant.png)


## 🛠️ Features

### 🔍 Retrieval-Augmented Generation (RAG) 
- Retrieves relevant chunks from uploaded course materials. 
- Uses FAISS vector search for fast similarity lookup.

### 🤖 Google Gemini LLM  
- Generates structured, high-quality English answers. 
- Includes improved prompting tuned for lecture/exam explanations.

### 📝 Document-Based Q&A
- Answers are only generated using your PDFs (slides, notes).
- Works for any course — simply replace the documents.

### 🌐 Web UI  
- Clean, simple HTML frontend for asking questions.
- Shows retrieved passages + relevance scores.
Runs fully online via Cloud Run.

### ☁️ Cloud Run Deployment 
- Backend downloads FAISS index & chunks from Google Cloud Storage at startup. 
- No local files needed on server.


## 📁 Project Structure

```pgsql
rag-study-assistant/
│
├── data/                  # Local document & index storage (dev only)
│   ├── *.pdf
│   ├── chunks.json
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   ├── faiss_metadata.json
│
├── rag/
│   ├── app.py             # FastAPI backend + Cloud Run startup logic
│   ├── llm_wrapper.py     # Prompting + Gemini API wrapper
│   ├── query_faiss.py     # Vector search over FAISS index
│   └── gcs_utils.py       # Download index from GCS
│
├── src/
│   ├── ingest.py          # Chunk PDFs → chunks.json
│   ├── embed_faiss.py     # Embed chunks → FAISS index
│   └── eval_rag.py
│
├── frontend/
│   └── index.html         # Web UI served via FastAPI `/web`
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .env
```

## Requirements

- Python 3.11+
- Google Cloud Account
- Gemini API Key
- Cloud Run enabled
- GCS bucket created

## Usage

### 1. Create `.env`

Create a `.env` file in the project root:

```ini
GEMINI_API_KEY=YOUR_KEY
GEMINI_MODEL_NAME=gemini-2.5-flash   # or another supported model
GCS_BUCKET_NAME=rag-documents-bucket-xxx
```


### 2. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```


### 3. Add Course Documents

Place your `.pdf` files into:
```kotlin
data/*.pdf
```

For example:
```bash
data/lecture1.pdf
data/lecture2.pdf
data/chapter5.pdf
```

 When switching courses, simply delete old PDFs and upload new ones.


### 4. Ingest PDFs → Chunks

```bash
python src/ingest.py
```

Generates:

```bash
data/chunks.json
```


### 5. Embed Chunks → FAISS Index

```bash
python src/embed_faiss.py
```

Generates & uploads to GCS:

```bash
data/faiss_index.bin
data/faiss_metadata.json
data/chunks.json
data/embeddings.npy
```


### 6. Run Backend Locally

```bash
uvicorn rag.app:app --reload
```

Endpoints:

- Swagger → http://127.0.0.1:8000/docs 
- Web UI → http://127.0.0.1:8000/web 
- Health → http://127.0.0.1:8000/health


### 7. Docker (Optional)

#### Build image:

```bash
docker build -t rag-app .
```

#### Run container:

```bash
docker run -p 8000:8000 --env-file .env rag-app
```


### 8. Deploy to Google Cloud Run

#### 8.1 Build & Push Docker Image:

```bash
gcloud builds submit \
  --tag europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/rag-repo/rag-app
```

#### 8.2 Deploy to Cloud Run:

```bash
gcloud run deploy rag-service \
  --image europe-west1-docker.pkg.dev/YOUR_PROJECT_ID/rag-repo/rag-app \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=YOUR_KEY,GEMINI_MODEL_NAME=gemini-2.0-flash,GCS_BUCKET_NAME=rag-documents-bucket-xxx
```

Cloud Run will output your service URL:

```bash
https://rag-service-xxxx-ew.a.run.app
```

You can now open:

```bash
https://rag-service-xxxx-ew.a.run.app/web
```


## Architecture - How the System Works
>#### 1. Student uploads PDFs (locally during ingestion)
>>PDFs → text → chunks.

>#### 2. Embeddings generated
>>Chunks → embeddings → FAISS index.

>#### 3. Index uploaded to GCS
>>Cloud Run always loads latest index.

>#### 4. FastAPI backend
>>Handles `/ask`:
>>>- retrieves top-k chunks
>>>- sends them to Gemini
>>>- returns structured English answer

>#### 5. Frontend UI
>>Shows:
>>>- Generated answer
>>>- Retrieval time
>>>- Passage list with scores

### Number of Passages (top-k)

In the web interface, you can choose how many document passages will be retrieved and used as context for Gemini.

- **top-k = 3** → fast, short answers  
- **top-k = 5** → recommended (balanced accuracy + speed)  
- **top-k = 7–10** → more detailed, lecture-style answers

Higher values include more slides but may increase response time.


![Rag Question & Answer](/assets/mapreduceoutput.png)
![Rag Chunks](/assets/mapreducechunks.png)


## Ideal Use Case
This system is perfect for a student who wants:

- A personal AI assistant for one specific course
- Answers only from their lecture materials
- Quick re-indexing when switching courses
- Cloud deployment with zero local dependencies

Examples:

- "How does Map-Reduce work step-by-step?"
- "Summarize Lecture 3."
- "What is virtualization in cloud computing?"
- "What are AWS IAM roles?"


## Notes
- All answers are generated in English.

- Answers are grounded in retrieved PDFs; if unrelated, model says so.

- You can expand the frontend to allow file upload to GCS (future feature).

- Works with any course: you control the documents.

## License

Distributed under the **MIT License**.
