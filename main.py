from fastapi import FastAPI, UploadFile, File, Form
from backmain import ResumeScreener, SentimentAnalyzer

app = FastAPI()
@app.post("/analyze_resume/")
async def analyze_resume_api(job_description: str = Form(...), file: UploadFile = File(...)):
    resume_screener=ResumeScreener()
    try:
        content = await file.read()
        resume_text = resume_screener.extract_text_from_pdf(content)
        result = resume_screener.analyze_resume(resume_text, job_description)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze_sentiment/")
async def analyze_sentiment_api(feedback: str = Form(...)):
    sentiment_analyzer = SentimentAnalyzer()  # ✅ Load here too
    sentiment = sentiment_analyzer.analyze_sentiment(feedback)
    score = sentiment["score"]
    recommendations = sentiment_analyzer.generate_recommendations(feedback, score)
    risk = sentiment_analyzer.extract_risk_from_text(recommendations)
    return {
        "sentiment": sentiment,
        "recommendations": recommendations,
        "risk": risk
    }
