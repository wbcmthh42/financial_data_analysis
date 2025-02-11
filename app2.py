import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from google import genai

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")  # New cache directory for markdown files
COMPANY_DOCS = {
    "Apple": "apple_FY24_report.pdf",
    "Alphabet": "Alphabet_FY24_report.pdf",
    "Microsoft": "MSFT_FY24Q4_10K.pdf"
}

def load_documents():
    """Load and convert all documents"""
    documents = {}
    converter = DocumentConverter()
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    for company, filename in COMPANY_DOCS.items():
        file_path = DATA_DIR / filename
        cache_path = CACHE_DIR / f"{company.lower()}_report.md"
        
        if file_path.exists():
            try:
                # Check if cached version exists
                if cache_path.exists():
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents[company] = content
                    st.sidebar.success(f"✓ {company} report loaded from cache")
                else:
                    # Convert and cache the document
                    doc = converter.convert(str(file_path))
                    markdown_content = doc.document.export_to_markdown()
                    
                    # Save to cache
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    
                    documents[company] = markdown_content
                    st.sidebar.success(f"✓ {company} report converted and cached")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {company} report: {str(e)}")
                continue
        else:
            st.sidebar.warning(f"⚠️ {company} report not found at {file_path}")
    return documents

def query_documents(question, documents):
    """Query the documents using Gemini"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Combine all documents' content
    context = " and ".join(documents.values())  # Updated since documents now contains markdown strings
    
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=f"You are an expert in financial analysis and charting. Context: {context}. {question}"
    )
    return response.text

# Streamlit UI
st.title("Financial Reports Analysis")

# Load documents
docs = load_documents()
st.sidebar.header("Loaded Documents")
for company in docs.keys():
    st.sidebar.success(f"✓ {company} report loaded")

# User input
user_question = st.text_area("Ask a question about the financial reports:", 
                            height=100,
                            placeholder="Example: What is the net cash after operations for Apple, Microsoft and Alphabet in 2023?")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Analyzing documents..."):
            answer = query_documents(user_question, docs)
            st.write("### Answer")
            st.write(answer)
    else:
        st.warning("Please enter a question")