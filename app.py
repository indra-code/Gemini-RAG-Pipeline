import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai
import os
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

def pdf_read(pdfs):
    text = ""
    for pdf in pdfs:
        docs = PdfReader(pdf)
        for doc in docs.pages:
            text+=doc.extract_text()
    return text

def chunk_split(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 15000, chunk_overlap = 1500)
    chunks = splitter.split_text(text)
    return chunks

def generate_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    store = FAISS.from_texts(chunks,embedding=embeddings)
    store.save_local("faiss.index")

def get_chain():
    prompt_template = '''
    Provide a detailed reply to the given question based on the given context.Your answer will only be based on the provided context and nothing else.If the answer to the question is not present in the context your reply should be 'Answer not in the provided context',\n\n
    Context : {context}\n\n
    Question: {question}
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt = PromptTemplate(input_variables=['context','question'],template=prompt_template)
    chain = load_qa_chain(model,chain_type='stuff',prompt= prompt)
    return chain

def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    data = FAISS.load_local("faiss.index",embeddings,allow_dangerous_deserialization=True)
    f_data = data.similarity_search(query)
    chain = get_chain()
    response = chain({'input_documents': f_data,'question': query},return_only_outputs=True)
    print(response)
    st.write("output: ",response['output_text'])

def main():
    st.set_page_config(page_title='Pdf reader')
    st.header('Generate answers from pdfs using Gemini')
    query = st.text_input("Ask your question")
    if query:
        user_input(query)
    with st.sidebar:
        st.title('Upload')
        pdfs = st.file_uploader("Upload files",accept_multiple_files=True)
        if st.button('Submit'):
            with st.spinner('Loading'):
                a = pdf_read(pdfs)
                b = chunk_split(a)
                generate_embeddings(b)
                st.success("Finished")
if __name__ == "__main__":
    main()




