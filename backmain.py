import nltk
import re
import pdfplumber
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import cohere

class ResumeScreener:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.cohere_api_key = "QU0eJVAl4MbACkDCy9WPN640qiViL1po6Z6kPr8S"
        self.cohere_client = None

    def extract_text_from_pdf(self, pdf_file):
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    def extract_keywords_from_job_description(self, job_description):
        tokens = nltk.word_tokenize(job_description.lower())
        keywords = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        return set(keywords)

    def extract_skills_from_resume(self, resume_text):
        if self.cohere_client is None:
            self.cohere_client = cohere.Client(self.cohere_api_key)

        prompt = f"""
        Extract all the skills from the following resume text. Only return a list of skills without any additional text:

        {resume_text}
        """

        response = self.cohere_client.generate(
            prompt=prompt,
            max_tokens=300,
            model="command-light",
            temperature=0.2
        )

        return response.generations[0].text.strip()

    def calculate_match_score(self, job_keywords, resume_skills):
        resume_skills_lower = resume_skills.lower()
        matches = sum(1 for keyword in job_keywords if keyword in resume_skills_lower)
        match_score = (matches / len(job_keywords)) * 100 if job_keywords else 0
        return round(match_score, 2)

    def analyze_resume(self, resume_text, job_description):
        job_keywords = self.extract_keywords_from_job_description(job_description)
        extracted_skills = self.extract_skills_from_resume(resume_text)
        match_score = self.calculate_match_score(job_keywords, extracted_skills)

        if self.cohere_client is None:
            self.cohere_client = cohere.Client(self.cohere_api_key)

        prompt = f"""
        Based on this resume text and job description:

        Resume text: {resume_text}

        Job description: {job_description}

        Provide an analysis of the candidate's suitability for the role. Focus on:
        1. Key strengths matching the job requirements
        2. Areas where the candidate may lack required skills
        3. Overall recommendation (Strongly Recommend, Recommend, Consider, Not Recommended)
        """

        response = self.cohere_client.generate(
            prompt=prompt,
            max_tokens=500,
            model="command-light",
            temperature=0.2
        )

        try:
            analysis = response.generations[0].text
        except:
            analysis = "Unable to analyze resume."

        return {
            "extracted_skills": extracted_skills,
            "match_score": match_score,
            "analysis": analysis
        }

class SentimentAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.cohere_api_key = "QU0eJVAl4MbACkDCy9WPN640qiViL1po6Z6kPr8S"
        self.cohere_client = None

    def analyze_sentiment(self, feedback):
        if self.sentiment_pipeline is None:
            self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

        result = self.sentiment_pipeline(feedback)
        return result[0]

    def generate_recommendations(self, feedback, sentiment_score):
        if self.cohere_client is None:
            self.cohere_client = cohere.Client(self.cohere_api_key)

        sentiment_label = "positive" if sentiment_score > 0.5 else "negative"
        prompt = f"""Based on the following employee feedback (classified as {sentiment_label}):"{feedback}"

        Provide a plain-text analysis with:
        - Key issues (if any)
        - Recommendations for HR
        - Estimated attrition risk (mention clearly: Low, Medium, or High)

        Keep the response clear and human-readable.
        """
        response = self.cohere_client.generate(
            prompt=prompt,
            max_tokens=500,
            model="command-light",
            temperature=0.2
        )
        try:
            return response.generations[0].text.strip()
        except:
            return "Unable to generate recommendations."

    def extract_risk_from_text(self, recommendations_text):
        risk = "Medium"
        lowered = recommendations_text.lower()
        if "attrition risk: low" in lowered or "low attrition risk" in lowered:
            risk = "Low"
        elif "attrition risk: high" in lowered or "high attrition risk" in lowered:
            risk = "High"
        elif "attrition risk: medium" in lowered or "medium attrition risk" in lowered:
            risk = "Medium"
        return risk
