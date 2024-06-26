### Project Overview

This project demonstrates the creation of a Natural Language Query Agent capable of answering questions based on a small set of lecture notes from Stanford's LLM lectures and a table of milestone LLM architectures. The system leverages LLMs and open-source vector indexing and storage frameworks to provide conversational answers, with an emphasis on follow-up queries and conversational memory. 

### Data Sources

1. **Stanford LLMs Lecture Notes**:
    - Introduction: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/introduction/)
    - Capabilities: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/capabilities/)
    - Harm-1: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/harm-1/)
    - Harm-2: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/harm-2/)
    - Data: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/data/)
    - Modeling: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/modeling/)
    - Training: [Lecture Link](https://stanford-cs324.github.io/winter2022/lectures/training/)
   
2. **Milestone Papers**: Table of model architectures from [Awesome LLM](https://github.com/Hannibal046/Awesome-LLM#milestone-papers).

### Project Structure

- **data_loading.py**: Contains functions to load data from the web and PDF.
- **processing.py**: Functions to split text into chunks and generate embeddings.
- **model_initialization.py**: Code to initialize the model and retrieval chain.
- **main.py**: Streamlit application for the chatbot interface.

### Intermediary Representation

**Data Organization and Embedding**:

1. **Raw Data Loading**: 
    - Web pages and PDF files are loaded using `WebBaseLoader` and `PyPDFLoader` respectively.
    
2. **Text Splitting**:
    - Documents are split into manageable chunks using `RecursiveCharacterTextSplitter` with a chunk size of 1200 characters and an overlap of 200 characters.
    
3. **Embedding**:
    - Text chunks are converted into embeddings using the HuggingFace model `BAAI/bge-small-en`.
    
4. **Vector Store**:
    - The embeddings are stored in a Chroma vector store, making them searchable.

### Detailed Steps

#### Loading Data

1. **WebBaseLoader**: Fetches and loads web pages.
2. **PyPDFLoader**: Loads and parses the PDF containing milestone papers.
3. **MergedDataLoader**: Merges the data from the web and PDF loaders.

#### Processing Data

1. **Text Splitting**: 
    - `RecursiveCharacterTextSplitter` divides the loaded text into smaller, overlapping chunks to ensure that context is preserved.
    
2. **Embedding Generation**:
    - `HuggingFaceBgeEmbeddings` generates embeddings for the text chunks using a pre-trained model.
    
3. **Vector Store**:
    - The Chroma vector store is used to store and index these embeddings, enabling efficient retrieval.

#### Initializing the Model

1. **LLM Initialization**:
    - `ChatGroq` initializes the chosen LLM model using the provided API key.
    
2. **Prompt Templates**:
    - Custom prompt templates are created to reformulate user queries and generate responses based on the retrieved context.
    
3. **Retrieval Chain**:
    - A retrieval chain is created that uses a history-aware retriever to provide context-aware answers.

### Application

A Streamlit application allows users to interact with the chatbot. Key features include:
- **Input Query**: Users can enter natural language queries.
- **Chat History**: The system maintains context across multiple queries.
- **Display of Sources**: The sources used to generate answers are displayed, ensuring transparency.

### Workflow of the System 
![ThP - Flowchart](https://github.com/wittyicon29/QABot-with-Conversational-Memory/assets/99320225/3af7d6a5-3628-4fed-915b-59e74077b31a)

### Deployment and Scaling

1. **Deployment Plan**:
    - Can be directly deployed over Streamlit Cloud for public access
    - Containerize the application using Docker for easy deployment.
    - Use cloud services like AWS or GCP for scalability.
    
3. **Scaling**:
    - Utilizing GPU capability to reduce the latency of generating the response.
    - As the number of lectures or papers grows, the retrieval can be made more efficient through improved vector storing
    - Implement caching strategies to improve response times for frequently asked questions.

### Improvements and Future Work

- **Enhanced Conversational Memory**: Improving the system's ability to handle complex, multi-turn conversations.
- **Citation and Reference Handling**: More sophisticated citation mechanisms to link specific sections of texts used in answers.

### Setup Instructions

1. **Clone the Repository**:
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```sh
    streamlit run main.py
    ```
