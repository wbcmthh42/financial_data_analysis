import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from openai import OpenAI
import json
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import chromadb
import numpy as np
import PyPDF2

import camelot
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

# Set up the Streamlit page
st.set_page_config(page_title="AI Learning Resources Assistant", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_pdf(file_path):
    """Process a single PDF file and return chunks"""
    converter = DocumentConverter()
    markdown_document = converter.convert(file_path)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    raw_chunks = splitter.split_text(markdown_document.document.export_to_markdown())
    
    # Add metadata to each chunk
    chunks = []
    for chunk in raw_chunks:
        chunk.metadata.update({
            "source": file_path,
            "content_type": "text"
        })
        chunks.append(chunk)

    return chunks

# def process_pdf(file_path):
#     """Process a single PDF file and return chunks"""
#     loader = PDFPlumberLoader(file_path)
#     pages = loader.load()
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=100,
#         separators=["\n\n", "\n", " ", ""],
#         length_function=len
#     )
    
#     chunks = text_splitter.split_documents(pages)
#     return chunks

# def process_pdf(file_path: str) -> List[Document]:
#     """Process a single PDF file and extract both tables and text"""
#     chunks = []
    
#     # Extract tables using Camelot
#     tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
#     for idx, table in enumerate(tables):
#         df = table.df
#         table_str = f"Table {idx + 1}:\n{df.to_string()}"
#         chunks.append(Document(
#             page_content=table_str,
#             metadata={
#                 "source": file_path,
#                 "page": table.parsing_report['page'],
#                 "content_type": "table",
#                 "table_number": idx + 1
#             }
#         ))
    
#     # Extract text using PyPDF2
#     with open(file_path, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
        
#         # Process each page
#         for page_num in range(len(pdf_reader.pages)):
#             page = pdf_reader.pages[page_num]
#             text = page.extract_text()
            
#             # Split text into chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=200,
#                 chunk_overlap=100,
#                 separators=["\n\n", "\n", " ", ""]
#             )
            
#             # Create Document objects for each text chunk
#             text_chunks = text_splitter.create_documents(
#                 texts=[text],
#                 metadatas=[{
#                     "source": file_path,
#                     "page": page_num + 1,
#                     "content_type": "text"
#                 }]
#             )
            
#             chunks.extend(text_chunks)
    
#     return chunks

@st.cache_resource(ttl="1h")
def initialize_vector_store():
    """Initialize and return the vector store"""
    persist_dir = "./chroma_db"
    
    try:
        # Delete existing database if it exists
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
        
        # Initialize ChromaDB client with settings
        settings = chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir,
            anonymized_telemetry=False
        )
        
        embeddings = OpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        
        vector_store = Chroma(
            collection_name="financial_docs",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        return vector_store
        
    except Exception as e:
        st.error(f"Failed to initialize vector store: {str(e)}")
        st.error("Please ensure the application has write permissions in the current directory.")
        return None

def list_available_pdfs():
    """List all PDF files in the data folder"""
    data_folder = 'vgc_data'
    
    if not os.path.exists(data_folder):
        st.error(f"Data folder '{data_folder}' not found!")
        return []
    
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    return pdf_files

def load_documents(selected_files=None):
    """Load and process selected PDFs from the data folder"""
    data_folder = 'vgc_data'
    
    if not os.path.exists(data_folder):
        st.error(f"Data folder '{data_folder}' not found!")
        return None
    
    # If no specific files are selected, use all PDF files
    if not selected_files:
        pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    else:
        pdf_files = selected_files
        
    if not pdf_files:
        st.warning(f"No PDF files selected to process!")
        return None
    
    all_chunks = []
    progress_bar = st.progress(0)
    
    for i, filename in enumerate(pdf_files):
        try:
            filepath = os.path.join(data_folder, filename)
            st.info(f"Processing {filename}...")
            chunks = process_pdf(filepath)
            all_chunks.extend(chunks)
            # Update progress bar
            progress_bar.progress((i + 1) / len(pdf_files))
            
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            continue
    
    return all_chunks

def query_llm(client, prompt, vector_store):

    def expand_query_with_llm(query):
        """Use LLM to generate query expansion terms."""
        expansion_prompt = f"""
        Expand the following financial query by providing synonyms, related concepts, and alternative phrasings. 
        Ensure the terms remain relevant to financial data analysis:
        
        Query: "{query}"
        
        Provide the expansion terms as a comma-separated list.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 or a suitable model
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=50,
            temperature=0  # Lower temp for more deterministic outputs
        )
        
        # Extract expansion terms
        expansion_terms = response.choices[0].message.content.strip().split(", ")
        
        # Combine with the original query
        expanded_query = query + " " + " ".join(expansion_terms)
        
        return expanded_query

    # Perform similarity search with a smaller k to avoid token limits    
    relevant_docs = vector_store.similarity_search(expand_query_with_llm(prompt), k=4)
    
    # Calculate token estimate (rough approximation)
    def estimate_tokens(text):
        # Approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    # Extract page_content from Document objects with token management
    context_parts = []
    total_tokens = 0
    token_limit = 100000  # Setting a conservative limit below the 128k max
    
    # Estimate tokens for system message and user prompt
    system_tokens = estimate_tokens(f"You are a financial data analyst assistant that helps analyze financial data from PDFs.")
    prompt_tokens = estimate_tokens(prompt)
    history_tokens = 0
    
    # Estimate tokens for chat history
    for message in st.session_state.messages:
        history_tokens += estimate_tokens(message["content"])
    
    # Calculate remaining tokens for context
    remaining_tokens = token_limit - system_tokens - prompt_tokens - history_tokens
    
    # Add docs until we approach the token limit
    for doc in relevant_docs:
        doc_tokens = estimate_tokens(doc.page_content)
        if (total_tokens + doc_tokens) > remaining_tokens:
            break
        context_parts.append(doc.page_content)
        total_tokens += doc_tokens
    
    context = "\n".join(context_parts)
    
    # Add a note if we had to limit context
    if len(context_parts) < len(relevant_docs):
        context += "\n\nNote: Some relevant information was omitted due to length constraints."
    
    messages = [
        {"role": "system", "content": f"""
        You are a financial data analyst assistant that helps analyze financial data from PDFs.
        
        Use this context from the documents to answer the question {expand_query_with_llm(prompt)}:
        {context}
        
        When analyzing the data:
        1. Look for specific financial information in the provided context
        2. If found, provide detailed analysis
        3. If not found, inform the user what information is available
        
        If a visualization would be helpful, include EXACTLY ONE Python code block with matplotlib code in this format:
        ```python
        import matplotlib.pyplot as plt
        
        # Data
        categories = ["cat1", "cat2", "cat3"]
        values = [val1, val2, val3]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        # ... plotting code ...
        plt.title("Chart Title")
        plt.xlabel("X Label")
        plt.ylabel("Y Label")
        plt.tight_layout()
        ```
        
        For monthly data, ensure to sort months chronologically, not alphabetically.
        Always include proper axis labels and titles.
        """}
    ]
    
    # Manage history to prevent token explosion
    # Only include the most recent conversation turns
    recent_history = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
    
    # Add conversation history and current prompt
    for message in recent_history:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            messages=messages,
            temperature=0
        )
        
        response_content = response.choices[0].message.content
    except Exception as e:
        st.error(f"Error querying the AI model: {str(e)}")
        return f"I'm sorry, I encountered an error processing your request. The error might be due to the large amount of document content. Try asking a more specific question or loading fewer documents."
    
    try:
        code_match = re.search(r'```python\s*(.*?)\s*```', response_content, re.DOTALL)
        
        if code_match:
            # Get the Python code
            plot_code = code_match.group(1)
            
            # Create a new figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a namespace with all required variables
            namespace = {
                'plt': plt,
                'np': np,
                'fig': fig,
                'ax': ax,
                'autolabel': lambda rects: [ax.annotate(f'{height:.0f}', 
                                                      xy=(rect.get_x() + rect.get_width()/2, height),
                                                      xytext=(0, 3),
                                                      textcoords='offset points',
                                                      ha='center') 
                                          for rect, height in [(rect, rect.get_height()) for rect in rects]]
            }
            
            # Execute the plot code in the namespace
            exec(plot_code, namespace)
            
            # Convert plot to image
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            
            # Store the new chart in session state
            st.session_state.charts.append(buf)
            
            # Remove the code block from the response
            response_content = re.sub(r'```python\s*.*?\s*```', '', response_content, flags=re.DOTALL)
            response_content = response_content.strip()
    
    except Exception as e:
        st.error(f"Error processing visualization: {str(e)}")
        st.exception(e)
    
    return response_content

def main():
    st.title("Financial Data Assistant")
    
    # Create tabs for Chat and Analysis
    tab1, tab2 = st.tabs(["Chat with Documents", "Document Analysis"])
    
    # Add sidebar with app information
    st.sidebar.title("Navigation")
    st.sidebar.info(
        "This app allows you to chat with your financial documents and perform analysis. "
        "Select files to analyze, load them into the system, then chat or analyze."
    )
    
    # Initialize charts list in session state if it doesn't exist
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    
    # Initialize loaded_files in session state if it doesn't exist
    if 'loaded_files' not in st.session_state:
        st.session_state.loaded_files = []
    
    # Load environment variables
    load_dotenv()
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Initialize vector store with better error handling
    try:
        vector_store = initialize_vector_store()
        if vector_store is None:
            st.error("Vector store initialization failed. Please check your settings and try again.")
            return
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        return
    
    # Document selection section
    st.sidebar.subheader("1. Select Documents")
    available_pdfs = list_available_pdfs()
    
    if available_pdfs:
        selected_files = st.sidebar.multiselect(
            "Select PDF files to analyze:",
            options=available_pdfs,
            default=None
        )
        
        st.sidebar.subheader("2. Load Documents")
        load_button = st.sidebar.button("Load Selected Documents")
        
        if load_button and selected_files:
            with st.spinner("Processing selected documents..."):
                try:
                    chunks = load_documents(selected_files)
                    if chunks:
                        try:
                            # Add documents in batches of 5000
                            batch_size = 5000
                            progress_bar = st.sidebar.progress(0)
                            total_batches = (len(chunks) - 1) // batch_size + 1
                            
                            for i in range(0, len(chunks), batch_size):
                                batch = chunks[i:i + batch_size]
                                vector_store.add_documents(batch)
                                progress = (i // batch_size + 1) / total_batches
                                progress_bar.progress(progress)
                                st.sidebar.info(f"Processed batch {i//batch_size + 1} of {total_batches}")
                            
                            st.sidebar.success(f"Successfully processed {len(chunks)} document chunks!")
                            # Store loaded files in session state
                            st.session_state.loaded_files = selected_files
                        except Exception as e:
                            st.sidebar.error(f"Error adding documents to vector store: {str(e)}")
                    else:
                        st.sidebar.warning("No documents were loaded. Please check your selection.")
                except Exception as e:
                    st.sidebar.error(f"Error processing documents: {str(e)}")
    else:
        st.sidebar.warning("No PDF files found in the data directory.")
    
    # Show currently loaded files
    if st.session_state.loaded_files:
        st.sidebar.subheader("Currently Loaded Files:")
        for file in st.session_state.loaded_files:
            st.sidebar.write(f"- {file}")
    
    # Chat management
    st.sidebar.subheader("3. Options")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.charts = []
        st.rerun()
    
    # CHAT TAB
    with tab1:
        st.subheader("Ask Questions About Your Financial Data")
        
        if not st.session_state.loaded_files:
            st.info("👈 Please select and load documents from the sidebar to start chatting.")
        
        # Display chat history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "viz" in message and message["viz"] is not None:
                    message["viz"].seek(0)
                    st.image(message["viz"].getvalue(), width=None)

        # Chat input - only enable if files are loaded
        if st.session_state.loaded_files:
            if prompt := st.chat_input("Ask me about the financial data!"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # Clear previous charts before getting new response
                    st.session_state.charts = []
                    
                    with st.spinner("Analyzing your question..."):
                        response = query_llm(client, prompt, vector_store)
                    st.markdown(response)
                    
                    # Display only the charts generated from this response
                    for chart in st.session_state.charts:
                        if chart is not None:
                            chart.seek(0)
                            st.image(chart.getvalue(), width=None)
                    
                    # Store both the response and the chart in the message history
                    message_with_viz = {
                        "role": "assistant", 
                        "content": response
                    }
                    if st.session_state.charts:  # Only add viz if there are charts
                        message_with_viz["viz"] = st.session_state.charts[0]
                    
                    st.session_state.messages.append(message_with_viz)
        else:
            st.chat_input("Load documents to start chatting...", disabled=True)
    
    # ANALYSIS TAB
    with tab2:
        st.subheader("Document Analysis")
        
        if not st.session_state.loaded_files:
            st.info("👈 Please select and load documents from the sidebar to perform analysis.")
            return
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            options=[
                "Executive Summary",
                "Key Financial Metrics",
                "Trend Analysis",
                "Risk Assessment",
                "Comparative Analysis"
            ]
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            run_analysis = st.button("Run Analysis", type="primary")
        
        if run_analysis:
            with st.spinner(f"Running {analysis_type}..."):
                # Construct a prompt based on the selected analysis type
                if analysis_type == "Executive Summary":
                    analysis_prompt = "Generate an executive summary of the key financial information in these documents. Include main insights, performance highlights, and important contextual information."
                
                elif analysis_type == "Key Financial Metrics":
                    analysis_prompt = "Extract and analyze the key financial metrics from these documents. Include revenue, profit margins, ROI, cash flow, debt ratios, and other important KPIs if available."
                
                elif analysis_type == "Trend Analysis":
                    analysis_prompt = "Analyze trends in the financial data over time. Identify patterns, growth rates, seasonal variations, and long-term trajectories for key financial indicators."
                
                elif analysis_type == "Risk Assessment":
                    analysis_prompt = "Perform a risk assessment based on the financial data. Identify potential financial risks, compliance issues, market exposure, and other factors that could impact financial stability."
                
                elif analysis_type == "Comparative Analysis":
                    analysis_prompt = "Compare financial performance across different periods, segments, or against industry benchmarks if such data is available in the documents."
                
                # Clear previous charts before getting new response
                st.session_state.charts = []
                
                # Query the LLM with the analysis prompt
                analysis_result = query_llm(client, analysis_prompt, vector_store)
                
                # Display results
                st.markdown("## Analysis Results")
                st.markdown(analysis_result)
                
                # Display charts if any were generated
                for chart in st.session_state.charts:
                    if chart is not None:
                        chart.seek(0)
                        st.image(chart.getvalue(), width=None)
                
                # Add a download button for the analysis report
                report = f"# {analysis_type} Report\n\n{analysis_result}"
                report_bytes = report.encode()
                st.download_button(
                    label="Download Analysis Report",
                    data=report_bytes,
                    file_name=f"{analysis_type.replace(' ', '_').lower()}_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
