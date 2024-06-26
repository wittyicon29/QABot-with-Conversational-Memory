from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

def process_data(docs_all):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(docs_all)
    
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    vectorstore = Chroma.from_documents(documents=docs_chunks, embedding=hf)
    return vectorstore
