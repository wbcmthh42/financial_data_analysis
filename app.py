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

def load_documents():
    """Load and process all PDFs from the data folder"""
    data_folder = 'data'
    
    if not os.path.exists(data_folder):
        st.error(f"Data folder '{data_folder}' not found!")
        return None
    
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    if not pdf_files:
        st.warning(f"No PDF files found in '{data_folder}' folder!")
        return None
    
    all_chunks = []
    for filename in pdf_files:
        try:
            filepath = os.path.join(data_folder, filename)
            st.info(f"Processing {filename}...")
            chunks = process_pdf(filepath)
            all_chunks.extend(chunks)
            
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            continue
    
    return all_chunks

def query_llm(client, prompt, vector_store):
    # Perform similarity search
    relevant_docs = vector_store.similarity_search(prompt, k=3)
    
    # Extract page_content from Document objects
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    messages = [
        {"role": "system", "content": f"""
        You are a financial data analyst assistant that helps analyze financial data from PDFs.
        
        Use this context from the documents to answer the question:
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
    
    # Add conversation history and current prompt
    for message in st.session_state.messages:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })
    
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        messages=messages,
        temperature=0
    )
    
    response_content = response.choices[0].message.content
    
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
    
    # Initialize charts list in session state if it doesn't exist
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    
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
    
    # Load documents button
    if st.sidebar.button("Load/Reload Documents"):
        with st.spinner("Processing documents..."):
            try:
                chunks = load_documents()
                if chunks:
                    try:
                        # Add documents in batches of 5000
                        batch_size = 5000
                        for i in range(0, len(chunks), batch_size):
                            batch = chunks[i:i + batch_size]
                            vector_store.add_documents(batch)
                            st.info(f"Processed batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}")
                        st.success(f"Successfully processed {len(chunks)} document chunks!")
                    except Exception as e:
                        st.error(f"Error adding documents to vector store: {str(e)}")
                else:
                    st.warning("No documents were loaded. Please check your data folder.")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "viz" in message and message["viz"] is not None:
                message["viz"].seek(0)
                st.image(message["viz"].getvalue(), width=None)

    # Chat input
    if prompt := st.chat_input("Ask me about the financial data!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Clear previous charts before getting new response
            st.session_state.charts = []
            
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

if __name__ == "__main__":
    main()
