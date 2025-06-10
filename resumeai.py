import streamlit as st
import pdfplumber
from google.generativeai import GenerativeModel
import google.generativeai as genai
import yaml
import os

# Configure Gemini API (replace with your API key)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyC_iRD_Ss1ayBXadgIIrHAoMKu2xoXkTFY"))

# Initialize Gemini model
model = GenerativeModel("gemini-1.5-flash")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Function to analyze resume using Gemini API
def analyze_resume(resume_text, job_description):
    prompt = f"""
    You are a smart AI assistant integrated into a resume analysis tool. Your job is to extract key skills, summarize professional experience, and evaluate the alignment between the candidate’s resume and a specified job role. You must be accurate, concise, and follow industry standards for resume assessment.

    Analyze the following resume content and perform the following tasks:

    1. Extract Key Skills: List technical, soft, and domain-specific skills found in the resume.
    2. Summarize Work Experience: Provide a 4–6 line summary of the candidate's professional background.
    3. Match with Job Role: Evaluate how well this resume aligns with the following job description. Give a match rating out of 10 and a brief justification.

    📄 Resume Content:
    {resume_text}

    🎯 Target Job Role/Description:
    {job_description}

    Output Format:
    ```yaml
    Key Skills:
    - Skill 1
    - Skill 2
    - Skill 3
    ...
    Summary of Experience:
    - [3–5 sentence professional summary]
    Alignment Rating:
    - Score: X/10
    - Justification: [2–3 sentence explanation of relevance, strengths, and gaps]
    ```
    """
    response = model.generate_content(prompt)
    return response.text.strip("```yaml\n").strip("```")

# Streamlit app
st.title("Resume Analyzer with Gemini API")
st.write("Upload a resume (PDF or TXT) and provide a job description to analyze the candidate's fit.")

# File upload
uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])

# Job description input
job_description = st.text_area("Enter Job Description", height=200)

# Analyze button
if st.button("Analyze Resume"):
    if uploaded_file is not None and job_description:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")

        # Get analysis from Gemini API
        with st.spinner("Analyzing resume..."):
            try:
                analysis = analyze_resume(resume_text, job_description)
                st.subheader("Analysis Results")
                st.code(analysis, language="yaml")

                # Parse and display YAML for better formatting
                parsed_yaml = yaml.safe_load(analysis)
                st.write("### Key Skills")
                for skill in parsed_yaml.get("Key Skills", []):
                    st.write(f"- {skill}")
                st.write("### Summary of Experience")
                st.write(parsed_yaml.get("Summary of Experience", ""))
                st.write("### Alignment Rating")
                st.write(f"**Score:** {parsed_yaml.get('Alignment Rating', {}).get('Score', 'N/A')}")
                st.write(f"**Justification:** {parsed_yaml.get('Alignment Rating', {}).get('Justification', 'N/A')}")
            except Exception as e:
                st.error(f"Error analyzing resume: {str(e)}")
    else:
        st.warning("Please upload a resume and provide a job description.")