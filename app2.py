import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from google import genai

# Load environment variables
load_dotenv()

# Setup paths
DATA_DIR = Path("vgc_data")
CACHE_DIR = Path("cache")  # Cache directory for markdown files

# Page config for better appearance
st.set_page_config(
    page_title="Financial Reports Analyzer",
    page_icon="📊",
    layout="wide"
)

def get_available_files():
    """Get list of available PDF files"""
    return [pdf_file.stem for pdf_file in DATA_DIR.glob("*.pdf")]

def load_document(company_name):
    """Load and convert a single document"""
    converter = DocumentConverter()
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True)
    
    pdf_file = DATA_DIR / f"{company_name}.pdf"
    cache_path = CACHE_DIR / f"{company_name.lower()}_report.md"
    
    try:
        # Check if cached version exists
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.sidebar.success(f"✓ {company_name} report loaded from cache")
            return content
        else:
            # Convert and cache the document
            doc = converter.convert(str(pdf_file))
            markdown_content = doc.document.export_to_markdown()
            
            # Save to cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            st.sidebar.success(f"✓ {company_name} report converted and cached")
            return markdown_content
    except Exception as e:
        st.sidebar.error(f"❌ Error loading {company_name} report: {str(e)}")
        return None

def query_documents(question, content):
    """Query the document using Gemini"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=f"You are an expert in financial analysis and charting. Context: {content}. {question}"
    )
    return response.text

# Streamlit UI
st.title("📊 Financial Reports Analysis")
st.markdown("---")

# Sidebar for document selection
st.sidebar.header("📁 Document Selection")

# Get available files
available_files = get_available_files()
if not available_files:
    st.sidebar.warning(f"⚠️ No PDF files found in {DATA_DIR}")
    st.warning(f"Please add PDF files to the {DATA_DIR} directory to analyze.")
else:
    # Single file selection dropdown
    selected_file = st.sidebar.selectbox(
        "Select a report to analyze:",
        options=available_files,
        format_func=lambda x: f"{x} Financial Report"
    )
    
    # Load button
    if st.sidebar.button("Load Selected Report"):
        with st.sidebar.status("Loading document..."):
            document_content = load_document(selected_file)
            if document_content:
                st.session_state.current_document = document_content
                st.session_state.current_company = selected_file
                st.sidebar.success(f"✅ {selected_file} report loaded successfully!")
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Query Report", "View Report Preview"])
    
    with tab1:
        st.header("Ask Questions About the Report")
        
        # Show currently loaded document
        if 'current_company' in st.session_state:
            st.info(f"Currently analyzing: **{st.session_state.current_company}** financial report")
            
            # User input
            user_question = st.text_area(
                "What would you like to know about this financial report?", 
                height=100,
                placeholder="Example: What is the net cash after operations for the company in 2024?"
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                submit = st.button("Get Answer", type="primary", use_container_width=True)
            
            if submit:
                if user_question:
                    with st.spinner("Analyzing document..."):
                        answer = query_documents(user_question, st.session_state.current_document)
                        st.write("### Answer")
                        st.markdown(answer)
                else:
                    st.warning("Please enter a question")
        else:
            st.warning("Please select and load a report from the sidebar first.")
            
    with tab2:
        st.header("Document Preview")
        if 'current_document' in st.session_state:
            with st.expander("Show Document Content", expanded=False):
                preview = st.session_state.current_document[:2000] + "..." if len(st.session_state.current_document) > 2000 else st.session_state.current_document
                st.markdown(preview)
        else:
            st.info("Load a document to see a preview here.")