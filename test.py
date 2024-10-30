# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:30:12 2024

@author: SunderAli.Khowaja
"""

import streamlit as st
import google.generativeai as genai
import time
import PyPDF2

# --- Configuration ---
genai.configure(api_key="AIzaSyAmKJP6jnY-pzKTTy2QecEXhL-MvZ4rO2o")

# Gemini Model initialization
MODEL_NAME = 'gemini-1.5-pro-001'

# --- Helper Functions ---
@st.cache_data
def extract_pdf_content(uploaded_file):
    """Extracts text content from the uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    content = ""
    for page in range(len(pdf_reader.pages)):
        content += "\n\n--- Page {} ---\n\n".format(page + 1)
        content += pdf_reader.pages[page].extract_text()
    return content

def role_to_streamlit(role):
    """Converts model role to Streamlit display format."""
    return "assistant" if role == "model" else role

def response_generator(text):
    """Generates response text with a typing effect asynchronously."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- Streamlit App ---
st.set_page_config(page_title="Research Paper Q&A Platform", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Research Paper")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    research_text = ""
    if uploaded_file is not None:
        research_text = extract_pdf_content(uploaded_file)
        st.success("PDF Uploaded Successfully!")

        # Preview limited content
        st.header("PDF Content Preview")
        st.text_area("Preview", research_text + "...", height=1200)


st.title("Research Paper Q&A Platform")

if uploaded_file and research_text:
    # Instruction text with embedded research content
    INSTRUCTIONS = f"""
    You are an expert researcher skilled in academic paper analysis. Given the research paper provided below, your task is to answer the question in a clear, concise, and evidence-based manner. Your response should:

    Thoroughly Review: Carefully examine key sections of the paper, including the abstract, introduction, methodology, results, and discussion.
    
    Understand the Question: Fully grasp the question being asked before searching for the answer in the paper.
    
    Provide a Structured Answer:
    
    Directly address the question.
    Reference specific sections of the paper (e.g., "According to the methodology section (p. 15)...").
    Include relevant data or findings from the paper.
    Use citations to support your response.
    Ensure Clarity and Precision: Deliver a clear, precise answer, avoiding unnecessary jargon. Aim to be informative and straightforward.
    
    Format Your Response:
    
    Summary of Findings: Briefly summarize key findings related to the question.
    Evidence from the Paper: Cite specific sections or data from the research paper that support your answer.
    Conclusion: Draw a concise conclusion based on the evidence provided.
    Below is the content of the research paper for your reference:
    
    <{research_text}>
    """

    # Lazy initialization of the model
    if "chat" not in st.session_state:
        model = genai.GenerativeModel(
            MODEL_NAME, 
            system_instruction=INSTRUCTIONS, 
            generation_config=genai.types.GenerationConfig(
                candidate_count=1, top_p=0, top_k=1, temperature=0.0
            )
        )
        st.session_state.chat = model.start_chat(history=[])

    # Display chat history
    for message in st.session_state.chat.history:
        with st.chat_message(role_to_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # User input
    if prompt := st.chat_input("Ask your question here..."):
        # Display user's question
        st.chat_message("user").markdown(prompt)

        # Generate and display response
        response = st.session_state.chat.send_message(prompt)
        with st.chat_message("assistant"):
            st.write_stream(response_generator(response.text))
else:
    st.info("Please upload a research paper to start.")
