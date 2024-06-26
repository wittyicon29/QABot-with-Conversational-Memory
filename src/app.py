import streamlit as st
from langchain_core.messages import HumanMessage
import os
from data_loading import load_data
from processing import process_data
from model_initialization import initialize_model

chat_history = []

def format_source(source):
    """Format the source information for better readability."""
    metadata = source.metadata
    title = metadata.get('title', 'Unknown Title')
    url = metadata.get('source', 'Unknown Source')
    return f"**Title**: {title}\n**URL**: {url}\n"

def display_chat_history(chat_history):
    """Display the chat history."""
    if not chat_history:
        st.warning("No chat history to display.")
        return
    
    st.subheader("Chat History")
    for i, msg in enumerate(chat_history):
        if isinstance(msg, HumanMessage):
            st.markdown(f"**Query {i//2 + 1}:** {msg.content}")
        else:
            st.markdown(f"**Answer {i//2 + 1}:** {msg.content}")
            
def summarize_conversation(chat_history):
    """Generate a summary of the conversation using the LLM."""
    if not chat_history:
        return "No conversation to summarize."
    
    summary_prompt = "Summarize the following conversation:\n\n"
    conversation = ""
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            conversation += f"User: {msg.content}\n"
        else:
            conversation += f"Assistant: {msg.content}\n"
    
    input_text = summary_prompt + conversation
    summary_result = rag_chain.invoke({"input": input_text})
    return summary_result["answer"]

st.title("RAG Chatbot with Conversational Memory")
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter your GROQ Cloud API key", type="password")

model_name = st.sidebar.selectbox(
    "Select Model",
    options=[
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "llama3-70b-8192"
    ]
)

if st.sidebar.button("Clear Chat History"):
    chat_history = []
    st.sidebar.success("Chat history cleared.")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    st.sidebar.success("API key set successfully")

    docs_all = load_data()
    vectorstore = process_data(docs_all)
    rag_chain = initialize_model(vectorstore, api_key, model_name)

    query = st.text_input("Enter your query:")

    if st.button("Ask"):
        if query:
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})
            answer = result["answer"]
            context = result["context"]

            st.markdown(f"**Answer:** {answer}")

            st.markdown("**Source(s):**")
            displayed_urls = set()
            for doc in context:
                metadata = doc.metadata
                url = metadata.get('source', 'Unknown Source')
                if url not in displayed_urls:
                    st.markdown(format_source(doc))
                    displayed_urls.add(url)

            chat_history.extend([HumanMessage(content=query), result["answer"]])

    if st.sidebar.button("Display Chat History"):
        display_chat_history(chat_history)
        
    if st.sidebar.button("Summarize Conversation"):
        summary = summarize_conversation(chat_history)
        st.markdown(f"**Conversation Summary:** {summary}")
else:
    st.warning("Please enter your API key in the sidebar to continue.")
