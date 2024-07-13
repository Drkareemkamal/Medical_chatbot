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
If you don't know the answer, please just say that don't know the answer, don't try to make uo an answer.
Context : {context}
Question : {question}
only return the helpful answer below and nothing else.
'''

def set_custom_prompt():
    """
    Prompt template for QA retrieval for vector stores
    """
    prompt = PromptTemplate(template = custom_prompt_template,
                            input_variables = ['context','question'])
    
    return prompt
    

def load_llm():
    llm = CTransformers(
        #model = 'TheBloke/Llama-2-7B-Chat-GGML',
        model = 'model\llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type = 'llama',
        max_new_token = 512,
        temperature = 0.5
    )
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs= {'k': 2}),
        return_source_documents = True,
        chain_type_kwargs = {'prompt': prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceBgeEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                          model_kwargs = {'device':'cpu'})
    
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})

    return response

def get_pdf_page_as_image(pdf_path, page_number):
    document = fitz.open(pdf_path)
    page = document.load_page(page_number)  
    pix = page.get_pixmap()
    img = Image.open(io.BytesIO(pix.tobytes()))
    return img

