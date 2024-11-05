"""

Kindly have all these libraries installed in your system - 

pip install langchain
pip install huggingface-hub
pip install sentence-transformers
pip install chromadb
pip install langchain-text-splitters
pip install unstructured[pdf]  
pip install langchain-community

"""

import streamlit as st
import time
import os
import requests
from backend import fetch_papers
from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader  
from langchain_core.prompts.prompt import PromptTemplate  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma  
from langchain_huggingface import HuggingFaceEndpoint

from langchain.chains import RetrievalQA  
from config import HUGGINGFACE_API_KEY  

def adjust_result(result, url):
    """
    Adjust the result to include the answer and append URLs for more details.

    Args:
        result (str): The initial result string.
        url (list): A list of URLs for further reading.

    Returns:
        str: The adjusted result with appended URLs.
    """
    left_pos = result.rfind("Answer")  # Find the position of "Answer" in the result
    right_pos = result.rfind(".")  # Find the position of the last period
    result = result[left_pos+7:right_pos+1]  # Extract the answer
    result += "\nFor more details visit:\n"
    for i in url:  # Append each URL to the result
        result += f"- {i}\n"
    return result

def store_data_into_vectorDB(query):
    """
    Fetch papers based on the query and store their data into a vector database.

    Args:
        query (str): The search query.

    Returns:
        tuple: A tuple containing URLs and the document search object.
    """
    url = fetch_papers(query)  # Fetch URLs of relevant papers
    if len(url) != 0:
        for i in url:
            try:
                loader = OnlinePDFLoader(i)  # Try loading the PDF from the URL
            except:
                continue  # Skip if there's an error loading the PDF
            pages = loader.load()  # Load the PDF pages
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)  # Split pages into chunks
            docs = splitter.split_documents(pages)  # Split the pages into documents
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Initialize embeddings
            doc_search = Chroma.from_documents(
                documents=docs,
                embedding=embeddings
            )  # Create a Chroma vector store
        return (url, doc_search)
    else:
        return ([], '')  # Return empty if no URLs found

def fetch_data(query):
    """
    Fetch data based on the query by utilizing the vector database and a language model.

    Args:
        query (str): The search query.

    Returns:
        str: The result from the language model or an error message.
    """
    url, doc_search = store_data_into_vectorDB(query)  # Store data into vector database
    if len(url) != 0:
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # Model repository ID
        llm = HuggingFaceEndpoint(huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                         repo_id=repo_id, temperature= 0.7, max_length= 1500)
        # Define the prompt templatestre
        template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the detail
            the answer is not in the
        provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n{context}?\n
        Question: \n{question}\n
        Answer:
        """
        prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
        )
        # Create a retrieval QA chain
        retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever(), 
            chain_type_kwargs={"prompt": prompt})
        result = retrieval_chain.invoke(query)['result']  # Invoke the chain with the query
        result = adjust_result(result, url)  # Adjust the result to include URLs
        
        return result



        # else:
            # return 'OOPS! Something went wrong'
    # except Exception:
        # return 'OOPS! Something went wrong'

def display_mainform():
    """
    Display the main form in the Streamlit app, allowing users to interact with the chatbot.
    """
    with st.sidebar:  # Display the sidebar
        with st.container():
            st.header('', divider='red')

            st.image(
                "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQDiJY1s_xnzXt0nZ1pRfWXNd5Ol4ggx3f7qA&s",
                width=None,
            )

    st.header('Chat With Academic Resources', divider='red')  # Display the main header

    def response(prompt):
        """
        Fetch response for the given prompt.

        Args:
            prompt (str): The user's input query.

        Returns:
            str: The response from the language model.
        """
        response = fetch_data(prompt)  # Fetch data for the prompt
        return response

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking_message = st.empty()
            thinking_message.markdown("Thinking...")
            # Fetch the response
            response = response(prompt)
            # Replace "Thinking..." with the actual response
            thinking_message.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    display_mainform()  # Run the main form display function
    
    

