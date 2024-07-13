
# Medical Chatbot

This project creates a Medical Chatbot using LangChain for document processing, HuggingFace for embeddings, and Streamlit for deployment. The chatbot reads the "Gale Encyclopedia of Medicine" PDF and provides answers to user queries by retrieving relevant information from the document.

!["This is a flowchart describing a simple local retrieval-augmented generation (RAG) workflow for document processing and embedding creation, followed by search and answer functionality. The process begins with a collection of documents, such as PDFs or a 1200-page nutrition textbook, which are preprocessed into smaller chunks, for example, groups of 10 sentences each. These chunks are used as context for the Large Language Model (LLM). A cool person (potentially the user) asks a query such as "What are the macronutrients? And what do they do?" This query is then transformed by an embedding model into a numerical representation using sentence transformers or other options from Hugging Face, which are stored in a torch.tensor format for efficiency, especially with large numbers of embeddings (around 100k+). For extremely large datasets, a vector database/index may be used. The numerical query and relevant document passages are processed on a local GPU, specifically an RTX 4090. The LLM generates output based on the context related to the query, which can be interacted with through an optional chat web app interface. All of this processing happens on a local GPU. The flowchart includes icons for documents, processing steps, and hardware, with arrows indicating the flow from document collection to user interaction with the generated text and resources."](images.png)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup Instructions](#setup-instructions)
3. [Ingest Pipeline](#ingest-pipeline)
4. [Model Pipeline](#model-pipeline)
5. [Streamlit Deployment](#streamlit-deployment)

## Project Overview

The Medical Chatbot project processes a PDF file containing medical information, splits it into smaller text chunks, creates embeddings for these chunks, and stores them in a vector database. The chatbot then uses this database to answer user queries by retrieving relevant information from the PDF. 

The project is divided into three main parts:
1. **Ingest Pipeline:** Loads and processes the PDF document.
2. **Model Pipeline:** Creates a question-answering model using the embeddings and a language model.
3. **Streamlit Deployment:** Deploys the chatbot using Streamlit for an interactive user interface.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Medical_chatbot.git
    
    ```
    ```bash
    cd Medical_chatbot
    ```
    
    ### Create environment
    
    ```bash
    python -m venv venv
    ```
    
    ### Activate environment
    
    Linux/macOS:
    ```
    source venv/bin/activate
    ```
    
    Windows: 
    ```bash
    .\venv\Scripts\activate
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the PDF file ("Gale_encyclopedia_of_medicine_vol_1.pdf") in the `data` folder.

## Ingest Pipeline

The ingest pipeline processes the PDF document and creates embeddings for the text chunks.

### ingest.py

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = 'vectorstores/'

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
```

**Explanation:**
1. **DirectoryLoader:** Loads the PDF documents from the specified directory.
2. **RecursiveCharacterTextSplitter:** Splits the documents into smaller chunks for processing.
3. **HuggingFaceBgeEmbeddings:** Converts the text chunks into embeddings using the specified model.
4. **FAISS:** Stores the embeddings in a vector database for efficient retrieval.

## Model Pipeline

The model pipeline sets up the language model and retrieval-based question-answering system.

### model.py

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.retrieval_qa.base import RetrievalQA
import fitz  # PyMuPDF
from PIL import Image
import io

DB_FAISS_PATH = 'vectorstores/'

custom_prompt_template = '''use the following pieces of information to answer the user's questions.
If you don't know the answer, please just say that don't know the answer, don't try to make up an answer.
Context : {context}
Question : {question}
only return the helpful answer below and nothing else.
'''

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        max_new_token=512,
        temperature=0.5
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

def get_pdf_page_as_image(pdf_path, page_number):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes()))
    return img
```

**Explanation:**
1. **PromptTemplate:** Defines the template for the question-answering prompt.
2. **CTransformers:** Loads the language model for generating responses.
3. **RetrievalQA:** Sets up the retrieval-based QA system using the embeddings and language model.
4. **FAISS:** Loads the vector database for retrieving relevant document chunks.
5. **get_pdf_page_as_image:** Converts a specific PDF page to an image for display.

## Streamlit Deployment

The Streamlit deployment provides an interactive web interface for the chatbot.

### app.py

```python
import os
import streamlit as st
from model import final_result, get_pdf_page_as_image

DB_FAISS_PATH = 'vectorstores/'
pdf_path = 'data/Gale_encyclopedia_of_medicine_vol_1.pdf'

st.title('Medical Chatbot')

user_query = st.text_input("Please enter your question:")

if st.button('Get Answer'):
    if user_query:
        response = final_result(user_query)
        if response:
            st.write("### Answer")
            st.write(response['result'])

            if 'source_documents' in response:
                st.write("### Source Document Information")
                for doc in response['source_documents']:
                    formatted_content = doc.page_content.replace("\n", "
")
                    st.write("#### Document Content")
                    st.text_area(label="Page Content", value=formatted_content, height=300)

                    source = doc.metadata['source']
                    page = doc.metadata['page']
                    st.write(f"Source: {source}")
                    st.write(f"Page Number: {page + 1}")

                    pdf_page_image = get_pdf_page_as_image(pdf_path, page)
                    st.image(pdf_page_image, caption=f"Page {page + 1} from {source}")
        else:
            st.write("Sorry, I couldn't find an answer to your question.")
    else:
        st.write("Please enter a question to get an answer.")
```

**Explanation:**
1. **User Input:** Takes the user's question as input.
2. **Button:** Triggers the retrieval and answering process.
3. **final_result:** Calls the function to get the answer and relevant document chunks.
4. **Display:** Shows the answer and source document details, including the PDF page as an image.

## Running the Application

1. Run the ingest pipeline to create the vector database:
    ```bash
    python ingest.py
    ```

2. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Conclusion

This project demonstrates the creation of a Medical Chatbot using advanced NLP and document retrieval techniques. The chatbot can efficiently process and retrieve information from a large PDF document, providing accurate answers to user queries.
