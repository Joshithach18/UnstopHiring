# 🧠 AI-Powered Resume Screener & Sentiment Analyzer API

This project provides an AI-powered API to:
- Analyze resumes and job descriptions
- Extract skills and match scores
- Assess candidate suitability
- Perform sentiment analysis on employee feedback
- Generate HR recommendations and attrition risk levels

Built with **FastAPI**, **Cohere NLP API**, and **Transformers**, it supports PDF parsing, natural language understanding, and scalable deployment as an API.

---

## 🚀 Features

- ✅ Extract text and skills from uploaded resumes (PDF)
- ✅ Match resumes to job descriptions using keyword analysis
- ✅ Generate detailed candidate evaluation and recommendation
- ✅ Perform sentiment analysis on employee feedback
- ✅ Extract HR insights and predict attrition risk

---

## 🛠️ Tech Stack

- **FastAPI** – Web API framework
- **Cohere API** – NLP generation (skills and analysis)
- **Transformers** – Sentiment classification with HuggingFace
- **pdfplumber** – PDF text extraction
- **NLTK** – Tokenization and stopword filtering
- **Plotly / Seaborn** – Data visualization (optional)
- **Uvicorn** – ASGI server for deployment

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## ▶️ Running the API
To start the FastAPI server locally:

```bash
uvicorn main:app --reload
```

Access interactive API docs at:

👉 http://127.0.0.1:8000/docs

## 📤 API Endpoints

1. /analyze_resume/ – Analyze Resume and Job Description
   
Method: POST

Request Type: multipart/form-data

Fields:

file: PDF file of the resume

job_description: Text description of the job role

Curl Example:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/analyze_resume/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'job_description=Looking for a Python Developer...' \
  -F 'file=@Resume.pdf;type=application/pdf'
```

Returns:

Extracted skills

Match score (0–100%)

AI-generated analysis and recommendation

2. /analyze_feedback/ – Analyze Employee Feedback

Method: POST

Body: {"feedback": "<your text here>"}

Returns:

Sentiment label and score

Recommendations for HR

Estimated attrition risk

## 🔐 Notes

This project uses the Cohere API for text generation. Replace cohere_api_key with your actual API key or use environment variables.

## 🙏 Acknowledgements

Cohere for free NLP API access

HuggingFace for sentiment models

FastAPI for an amazing web framework

Make sure to download NLTK stopwords and punkt tokenizer when running the code for the first time.


---

Let me know if you'd like to customize the README further — for deployment on Render, Docker setup, Streamlit frontend integration, etc.
