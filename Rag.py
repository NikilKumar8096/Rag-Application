import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from google.api_core import exceptions

# Setting up Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDgAYfuR-WOVWrnHTc6KfGBqRVO7ELFng4"  # Replace with your actual API key

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Streamlit app
st.title("Magical Potion Shopping Assistant")
st.write("Upload a PDF with potion details and ask about potions!")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload Potion Catalog (PDF)", type="pdf")

# Initialize session state for vector store and QA chain
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Process PDF and create vector store
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing potion catalog..."):
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(pdf_text)
        
        # Create embeddings
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Create FAISS vector store
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_store = vector_store
            
            # Set up LLM and QA chain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            
            prompt_template = """You are a magical potion shopping assistant. Use the provided context to give accurate and helpful recommendations about potions. If the user asks about a specific potion or effect, provide details from the context. If no relevant information is found, say so politely. Keep responses concise and magical in tone.

Context: {context}

User Query: {question}

Answer: """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt}
            )
            st.success("Potion catalog processed! Ask away!")
        except exceptions.InvalidArgument as e:
            st.error(f"Error processing catalog: {str(e)}. Please check your API key and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}. Please try again.")

# Chat interface
st.subheader("Ask About Potions")
user_query = st.text_input("What potion are you looking for? (e.g., 'potion for strength' or 'uses of moonlight elixir')")

if user_query and st.session_state.qa_chain:
    with st.spinner("Consulting the magical archives..."):
        try:
            # Get response from QA chain
            response = st.session_state.qa_chain({"query": user_query})
            answer = response["result"]
            
            # Display response
            st.write("**Magical Answer:**")
            st.write(answer)
            
            # Update chat history
            st.session_state.chat_history.append({"query": user_query, "answer": answer})
            
            # Display chat history
            st.subheader("Recent Inquiries")
            for chat in st.session_state.chat_history[-5:]:  # Show last 5 interactions
                st.write(f"**You asked:** {chat['query']}")
                st.write(f"**Response:** {chat['answer']}")
                st.write("---")
                
        except exceptions.InvalidArgument as e:
            st.error(f"Error generating response: {str(e)}. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}. Please try again.")
elif user_query and not st.session_state.qa_chain:
    st.warning("Please upload a potion catalog PDF first!")

# Instructions
st.sidebar.header("How to Use")
st.sidebar.write("1. Upload a PDF containing potion details.")
st.sidebar.write("2. Ask questions about potions or their effects.")
st.sidebar.write("3. Get magical recommendations based on the catalog!")
st.sidebar.write("Note: Ensure your Google API key is set in the code and required packages are installed.")