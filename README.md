
# ResumeAnalyzer and Career Assistance Using LLM

**ResumeAnalyzer** is an advanced AI-powered project that simplifies resume analysis. It extracts information such as name, skills, education, from resumes in **PDF format** using **OCR and NLP techniques**. Additionally, it calculates an **ATS (Applicant Tracking System) score** by comparing the applicant's skills to the job description using specialized algorithms.  

The project also includes a chatbot powered by **Llama 3.2 via Ollama**, which offers personalized guidance to help users improve their resumes and skills for better job prospects.  

---

## Features  
- Text extraction from resumes using OCR.  
- Categorization of extracted information into predefined fields (e.g., name, skills, education).  
- ATS score calculation using:  
  - **Levenshtein Distance**  
  - **Jaccard Similarity**  
  - **Semantic Similarity** via transformer models.  
- A chatbot feature using **Llama 3.2** to engage users with suggestions for improvement.  

---

## How We Integrated Llama 3.2 via Ollama  
We utilized **Llama 3.2** by leveraging the **Ollama API**, making it simple to interact with the model. Here's how it works:  

1. **Install Ollama**:  
   - Visit [Ollama](https://ollama.ai) and follow their instructions to install the platform for your OS.  

2. **Authenticate Ollama**:  
   - Log in and ensure your **API key** is configured for use.  

3. **Integrate with Python**:  
   - The chatbot feature interacts with **Llama 3.2** through the Ollama API using `requests`. Example code:  
     ```python
     import requests

     url = "http://localhost:11434/api/chat"
     payload = {
         "model": "llama3.2",
         "messages": [{"role": "user", "content": "How can I improve my resume for job Y?"}],
     }
     response = requests.post(url, json=payload)
     print(response.json()["content"])
     ```  

---

## Requirements  

Before running the project, ensure all dependencies are installed. Use the provided `requirements.txt` file:  

```bash
pip install -r requirements.txt
```  

**Key dependencies**:  
- `pdfplumber` for extracting text from PDFs.  
- `spacy` for keyword extraction and named entity recognition.  
- `transformers` for semantic similarity tasks.  
- `levenshtein` for distance-based string matching.  
- `streamlit` for building the UI.  
- `requests` for API interaction with Ollama.  

---

## How to Use This Repo  

### Fork the Repository  
1. Log in to GitHub.  
2. Click the **Fork** button in the top-right corner of this repo to create your own copy.  

### Clone the Repository  
1. Open your terminal.  
2. Clone the repository to your local machine:  
   ```bash
   git clone https://github.com/Pradeesh1108/resume-analyzer-and-career-assistance-using-LLM.git
   ```  
3. Navigate to the project folder:  
   ```bash
   cd resume-analyzer
   ```  

---
