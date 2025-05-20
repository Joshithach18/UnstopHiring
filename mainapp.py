import streamlit as st
from backmain import ResumeScreener, SentimentAnalyzer
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="HR Tech Solution", layout="wide")
    
    st.title("HR-Tech Innovation Solution")
    
    tab1, tab2 = st.tabs(["Resume Screening", "Employee Sentiment Analysis"])
    
    with tab1:
        st.header("AI-Powered Resume Screening")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_description = st.text_area("Enter the job description for Software Engineer role:", height=200,
                                         value="""Enter here...""")
        
        with col2:
            uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
        
        if uploaded_file is not None and job_description:
            resume_screener = ResumeScreener()
            
            with st.spinner("Analyzing resume..."):
                resume_text = resume_screener.extract_text_from_pdf(uploaded_file)
                results = resume_screener.analyze_resume(resume_text, job_description)
                
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Match Score")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=results["match_score"],
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Match %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    st.markdown("### Extracted Skills")
                    st.write(results["extracted_skills"])
                    
                with col2:
                    st.markdown("### Candidate Analysis")
                    
                    st.write(results["analysis"])
    
    with tab2:
        st.header("Employee Sentiment Analysis")
        
        sample_data = {
            "feedback": [
                "I enjoy working here. The team is great and management is supportive.",
                "My manager doesn't listen to my ideas. I feel undervalued.",
                "The work-life balance is great, but compensation could be better.",
                "I'm considering leaving because there's no clear growth path.",
                "The company culture is toxic and people are overworked."
            ]
        }
        
        feedback_options = ["Enter custom feedback", "Use sample feedback"]
        feedback_choice = st.radio("Choose an option:", feedback_options)
        
        if feedback_choice == "Enter custom feedback":
            feedback = st.text_area("Enter employee feedback:", height=150)
        else:
            feedback = st.selectbox("Select a sample feedback:", sample_data["feedback"])
        
        if feedback:
            sentiment_analyzer = SentimentAnalyzer()
            
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = sentiment_analyzer.analyze_sentiment(feedback)
                sentiment_label = sentiment_result["label"]
                sentiment_score = sentiment_result["score"]
                
                if sentiment_label == "positive":
                    normalized_score = sentiment_score
                elif sentiment_label == "neutral":
                    normalized_score = 0.5
                else:  # negative
                    normalized_score = 1 - sentiment_score
                
                recommendations = sentiment_analyzer.generate_recommendations(feedback, normalized_score)
                
                st.subheader("Analysis Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Sentiment Analysis")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=normalized_score * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Sentiment: {sentiment_label.capitalize()}"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    st.markdown("### Attrition Risk")
                    
                    risk = sentiment_analyzer.extract_risk_from_text(recommendations)
                    if risk == "Low":
                        st.success(f"Attrition Risk: {risk}")
                    elif risk == "Medium":
                        st.warning(f"Attrition Risk: {risk}")
                    else:
                        st.error(f"Attrition Risk: {risk}")
                    

                
                with col2:
                    st.markdown("### Full Analysis")
                    st.write(recommendations)
                        

if __name__ == "__main__":
    main()