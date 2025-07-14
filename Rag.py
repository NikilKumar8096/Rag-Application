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
os.environ["GOOGLE_API_KEY"] = "AIzaSyDgAYfuR-WOVWrnHTc6KfGBqRVO7ELFng4"  # Replace with your actual key

#  Function used to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

#  Title for Streamlit App
st.set_page_config(page_title="Potion Chat Assistant", page_icon="🧪")
st.title("🧪 Magical Potion Shopping Assistant")
st.write("Upload a PDF with potion details and chat with your magical assistant!")

#  Initializing the Session State 
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF and Build Vector Store 
uploaded_file = st.file_uploader("📜 Upload Potion Catalog (PDF)", type="pdf")

if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing your magical scroll... 🧙‍♂️"):
        try:
            # Extracting and splitting the text
            pdf_text = extract_text_from_pdf(uploaded_file)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(pdf_text)

            # Generate the embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(chunks, embeddings)
            st.session_state.vector_store = vector_store

            # Creating LLM and QA chain
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

            prompt_template = """You are a magical potion shopping assistant. Use the provided context to give accurate and helpful recommendations about potions. If the user asks about a specific potion or effect, provide details from the context. If no relevant information is found, say so politely. Keep responses concise and magical in tone.

Context: {context}

User Query: {question}

Answer: """
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": prompt}
            )
            st.session_state.qa_chain = qa_chain

            st.success("Your magical catalog is ready! Ask away. 🧪")
        except exceptions.InvalidArgument as e:
            st.error(f"Invalid API Key or embedding issue: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Chat Interface 
if st.session_state.qa_chain:
    # Initial greeting
    if len(st.session_state.chat_history) == 0:
        with st.chat_message("assistant"):
            st.write("Welcome, magical shopper! 🧙‍♀️ Ask me about potions from your uploaded catalog.")

    # Showing previous chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["message"])

    # Getting the user input
    user_input = st.chat_input("Ask about a potion or its effect...")

    if user_input:
        # Saving  and displaying user message
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Getting assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consulting magical scrolls..."):
                try:
                    result = st.session_state.qa_chain({"query": user_input})
                    answer = result["result"]
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "message": answer})
                except Exception as e:
                    st.error(f"Failed to retrieve magical wisdom: {e}")
else:
    st.info("Please upload a potion catalog PDF to begin chatting. 🧾")

# Sidebar Instructions 
st.sidebar.header("🧙 How to Use")
st.sidebar.markdown("""
1. Upload a **PDF** containing potion data.
2. Ask your magical assistant about potion **uses, effects, or recommendations**.
3. Watch as the assistant responds in a **messaging-style chat**.
4. PDF is processed only once per session.
""")
st.sidebar.caption("Powered by LangChain, Gemini, and FAISS ✨")