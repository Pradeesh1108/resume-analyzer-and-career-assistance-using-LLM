import streamlit as st
import ollama
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import tempfile
import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pdfplumber
import torch

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_skills' not in st.session_state:
    st.session_state.user_skills = None

if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False

# Page config
st.set_page_config(page_title="Resume Analyzer & Career Assistant", layout="wide")


# Initialize BERT model and tokenizer
@st.cache_resource
def load_bert_model():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_bert_model()


# PDF Text Extraction Functions
def extract_text_from_pdf_upload(uploaded_file):
    """
    Extract text from an uploaded PDF file using pdfplumber
    """
    text = ""
    try:
        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.seek(0)

            # Extract text using pdfplumber
            with pdfplumber.open(tmp_file.name) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""
    finally:
        # Clean up the temporary file
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

    return text.strip()


def extract_text_from_pdf_ocr(uploaded_file):
    """
    Extract text from an uploaded PDF file using PDF2Image and PyTesseract
    """
    text = ""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file.seek(0)

            # Convert PDF to images
            images = convert_from_path(tmp_file.name)

            # Extract text from each image
            for image in images:
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                text += pytesseract.image_to_string(gray) + "\n"
    except Exception as e:
        st.error(f"Error extracting text using OCR: {str(e)}")
        return ""
    finally:
        # Clean up the temporary file
        if 'tmp_file' in locals():
            os.unlink(tmp_file.name)

    return text.strip()


def extract_text_from_pdf(uploaded_file):
    """
    Try both text extraction methods and return the best result
    """
    # Try pdfplumber first
    text = extract_text_from_pdf_upload(uploaded_file)

    # If pdfplumber fails or returns empty text, try OCR
    if not text.strip():
        text = extract_text_from_pdf_ocr(uploaded_file)

    return text.strip()


# ATS Scoring Functions
def sliding_window_chunk(text, max_length=512, stride=256):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) > 10:  # Only keep chunks with meaningful content
            chunks.append(chunk)
        if i + max_length >= len(tokens):
            break
    return chunks


def compute_similarity_with_window(job_description, resume_text):
    job_description_chunks = sliding_window_chunk(job_description)
    resume_text_chunks = sliding_window_chunk(resume_text)

    if not job_description_chunks or not resume_text_chunks:
        return 0

    total_similarity = 0
    count = 0

    for job_chunk in job_description_chunks:
        for resume_chunk in resume_text_chunks:
            inputs = tokenizer(
                tokenizer.decode(job_chunk),
                tokenizer.decode(resume_chunk),
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            scores = outputs.logits.numpy()
            probabilities = np.exp(scores) / np.sum(np.exp(scores))
            total_similarity += probabilities[0][1]
            count += 1

    return total_similarity / count if count > 0 else 0


# Resume Parsing Functions
def parse_resume(text):
    model = "llama3.2:latest"

    # Extract the name
    name_query = """
    Please extract only the name of the person from the text. 
    Do not provide any other details such as title, contact information, or address. 
    Return only the person's full name, with no additional text.
    """

    name_response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': name_query + "\n" + text
        }
    ])
    name = name_response['message']['content'].strip()

    # Extract degrees pursued
    degree_query = """
    What are the degrees pursued by the person in this text ?
    don't forget to add the domain and it is a strict rules.Take only Bachelors or Masters degree and strictly don't add anyother text 
    and dont take school education also don't mention that he pursued masters or not
    """

    degree_response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': degree_query + "\n" + text
        }
    ])
    degrees = degree_response['message']['content'].strip()

    return {
        'Name': name,
        'Degrees': degrees
    }


def extract_skills_from_resume(resume_text):
    model = "llama3.2:latest"
    skills_query = """
    Please extract a comma-separated list of technical skills and competencies from the text.
    Only include hard skills, technologies, tools, and specific competencies.
    Format them as a simple comma-separated list without any additional text or categories.
    """

    skills_response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': skills_query + "\n" + resume_text
        }
    ])
    return skills_response['message']['content'].strip()


# Chatbot Interface
def chat_interface():
    st.subheader("Career Assistant Chatbot")
    st.write("Ask questions about your resume, job fit, or career advice!")

    # Initialize the chat if not started
    if not st.session_state.conversation_started and st.session_state.user_skills:
        initial_prompt = f"""
        You are an AI career assistant. A user has provided their skills and a job description. 
        Your role is to guide them through a conversation, assessing how their skills match the job description, 
        suggesting improvements, and answering their questions about skill enhancement or job fit.

        Job Description:
        {st.session_state.job_description}

        User's Skills:
        {st.session_state.user_skills}

        Start by giving a brief, friendly analysis of how their skills align with the job description.
        """

        response = ollama.chat(model="llama3.2:latest", messages=[
            {
                'role': 'system',
                'content': initial_prompt
            }
        ])

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response['message']['content'].strip()
        })
        st.session_state.conversation_started = True

    # Display chat history (show both user and assistant messages)
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")

    # Chat input (user's query)
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Immediately add user's message to the chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })

        # Prepare conversation history for Llama
        llama_messages = [
            {
                'role': 'system',
                'content': f"""You are an AI career assistant helping with job application advice.
                Job Description: {st.session_state.job_description}
                User's Skills: {st.session_state.user_skills}"""
            }
        ]
        llama_messages.extend(st.session_state.chat_history)

        # Get assistant response
        response = ollama.chat(model="llama3.2:latest", messages=llama_messages)

        # Add assistant response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response['message']['content'].strip()
        })

        # Rerun to update chat display
        st.rerun()


# Main App Layout
st.title("Resume ATS Analyzer & Career Assistant")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Resume Analysis", "Career Assistant"])

with tab1:
    # Resume analysis UI
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    with col2:
        st.subheader("Enter Job Description")
        job_description = st.text_area("Paste the job description here", height=200)

    if uploaded_file is not None and job_description:
        progress_bar = st.progress(0)

        # Add Start Analysis button
    if st.button("Start Analyzing"):
        if uploaded_file is not None and job_description:
            progress_bar = st.progress(0)

            with st.spinner('Processing resume and computing ATS score...'):
                try:
                    # Extract text and process resume
                    progress_bar.progress(20)
                    resume_text = extract_text_from_pdf(uploaded_file)

                    progress_bar.progress(40)
                    parsed_info = parse_resume(resume_text)

                    # Extract skills for chatbot
                    progress_bar.progress(60)
                    st.session_state.user_skills = extract_skills_from_resume(resume_text)
                    st.session_state.job_description = job_description

                    # Compute ATS score
                    progress_bar.progress(80)
                    ats_score = compute_similarity_with_window(job_description, resume_text)

                    # Display results
                    progress_bar.progress(100)
                    st.success("Analysis completed successfully!")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #000000; text-align: center;">
                            <strong>Extracted Name</strong>
                            <hr>
                            <p>{parsed_info['Name']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # For Education Details
                    with col2:
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #000000; text-align: center;">
                            <strong>Education Details</strong>
                            <hr>
                            <p>{parsed_info['Degrees']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # For ATS Compatibility Score
                    with col3:
                        score_percentage = float(ats_score * 100)
                        color = "green" if score_percentage >= 70 else "orange" if score_percentage >= 50 else "red"
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; background-color: #000000; text-align: center;">
                            <strong>ATS Compatibility Score</strong>
                            <hr>
                            <div style='font-size: 24px; color: {color};'>
                                {score_percentage:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Score interpretation
                    st.subheader("Score Interpretation")
                    if score_percentage >= 70:
                        st.success("Your resume is well-matched with the job description! ✨")
                    elif score_percentage >= 50:
                        st.warning("Your resume could use some improvements to better match the job description.")
                    else:
                        st.error("Your resume needs significant modifications to match the job requirements.")

                    # Extracted Skills
                    st.subheader("Extracted Skills")
                    st.write(st.session_state.user_skills)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    else:
        st.info("Please upload a PDF resume and enter a job description to begin analysis")

with tab2:
    if st.session_state.user_skills and st.session_state.job_description:
        chat_interface()
    else:
        st.info("Please complete the resume analysis first to enable the career assistant.")

st.markdown("---")