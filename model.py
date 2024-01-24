import os
import torch
import json
import re
from google.cloud import storage
from langchain.schema.document import Document
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, TextStreamer, pipeline
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from google.cloud import aiplatform
from langchain.chat_models import ChatVertexAI

RAG_FORMAT = """
{context}

Question: {question}
"""
SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. Each file consists of a summary of a episode from the Show Game of thrones. Each Season has 10 episodes. If you don’t know the answer, just say that you don’t know, don’t try to make up an answer."

def setup_environment():
    """
    Set up the environment by configuring Google Cloud credentials and determining the device (CPU or GPU).
    
    Returns:
        str: Device to be used, either "cuda:0" for GPU or "cpu" for CPU.
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_api.json'
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def load_documents(directory_path):
    """
    Load documents from a specified directory.
    
    Args:
        directory_path (str): Path to the directory containing documents.
    
    Returns:
        list: List of Document objects, each representing a document.
    """
    docs = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='latin-1') as file:
            content = file.read()
            docs.append(Document(page_content=content))
    return docs

def initialize_embeddings(device):
    """
    Initialize Hugging Face Instruct embeddings.
    
    Args:
        device (str): Device to use, either "cuda:0" for GPU or "cpu" for CPU.
    
    Returns:
        HuggingFaceInstructEmbeddings: Initialized embeddings model.
    """
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": device})

def split_text_documents(docs):
    """
    Split documents into smaller chunks of text using RecursiveCharacterTextSplitter.
    
    Args:
        docs (list): List of Document objects representing the documents.
    
    Returns:
        list: List of text chunks obtained by splitting the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=64)
    return text_splitter.split_documents(docs)

def create_chroma_database(texts, embeddings):
    """
    Create a Chroma database from text chunks and embeddings.
    
    Args:
        texts (list): List of text chunks.
        embeddings (HuggingFaceInstructEmbeddings): Initialized embeddings model.
    
    Returns:
        Chroma: Chroma database created from the input texts and embeddings.
    """
    return Chroma.from_documents(texts, embeddings, persist_directory="db")



DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

def generate_prompt(prompt,system_prompt=DEFAULT_SYSTEM_PROMPT) -> str:
    """
    Generate a formatted prompt for the Retrieval QA system.

    Args:
        prompt (str): RAG format prompt template.
        issue_type (str): Type of the issue for which the prompt is generated.

    Returns:
        str: Formatted prompt.
    """
    return f"""
            [INST] <<SYS>>
            {system_prompt}
            <</SYS>>
        {prompt} [/INST]
        """.strip()


def initialize_conversation_memory(memory_key: str, output_key: str, return_messages: bool) -> ConversationBufferMemory:
    """
    Initialize conversation memory for the assistant.
    
    Args:
        memory_key (str): Key for storing conversation history in memory.
        output_key (str): Key for storing the assistant's answer in memory.
        return_messages (bool): Flag indicating whether to return conversation messages.
    
    Returns:
        ConversationBufferMemory: Initialized conversation memory.
    """
    return ConversationBufferMemory(memory_key=memory_key, output_key=output_key, return_messages=return_messages)

def initialize_chat_model(model_name: str) -> ChatVertexAI:
    """
    Initialize the chat model for the assistant.
    
    Args:
        model_name (str): Name of the chat model to be used.
    
    Returns:
        ChatVertexAI: Initialized chat model.
    """
    return ChatVertexAI(model=model_name)

def initialize_retrieval_chain(llm_model, chain_type, retriever, get_chat_history, memory, return_generated_question, verbose, combine_docs_chain_kwargs):
    """
    Initialize the conversational retrieval chain for the assistant.
    
    Args:
        llm_model: Language model for chat responses.
        chain_type (str): Type of the retrieval chain.
        retriever: Document retriever for retrieving relevant information.
        get_chat_history: Function to get chat history.
        memory: Conversation memory.
        return_generated_question (bool): Flag indicating whether to return the generated question.
        verbose (bool): Flag indicating whether to print verbose information.
        combine_docs_chain_kwargs: Additional keyword arguments for combining documents.
    
    Returns:
        ConversationalRetrievalChain: Initialized conversational retrieval chain.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        chain_type=chain_type,
        retriever=retriever,
        get_chat_history=get_chat_history,
        memory=memory,
        return_generated_question=return_generated_question,
        verbose=verbose,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs
    )


def setup_and_initialize():
    """
    Set up and initialize the environment, load documents, create embeddings, and initialize the conversational chain.

    Returns:
        tuple: Tuple containing the initialized objects (DEVICE, db, qa_chain).
    """
    DEVICE = setup_environment()
    directory_path = 'data'
    docs = load_documents(directory_path)
    embeddings = initialize_embeddings(DEVICE)
    texts = split_text_documents(docs)
    db = create_chroma_database(texts, embeddings)
    template = generate_prompt(RAG_FORMAT,SYSTEM_PROMPT)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    memory = initialize_conversation_memory(memory_key="chat_history", output_key='answer', return_messages=False)
    model = initialize_chat_model(model_name="gemini-pro")
    qa_chain = initialize_retrieval_chain(
        llm_model=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        get_chat_history=lambda o: o,
        memory=memory,
        return_generated_question=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain

# Usage
