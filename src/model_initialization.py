from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

def initialize_model(vectorstore, api_key, model_name):
    """Initialize the LLM and create the prompt template and RAG chain."""
    llm = ChatGroq(model=model_name, groq_api_key=api_key)

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, vectorstore.as_retriever(), contextualize_q_prompt
    )
    
    system_prompt = """
        You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to construct a well-structured answer \
        to the question. If you don't know the answer, say that you don't know. \
        Make sure to summarize the answer where mentioned. \
        Construct an answer which would be easier to read for the user.
        
        {context}
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain
