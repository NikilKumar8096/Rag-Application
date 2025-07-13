# Enchanted Emporium Gen-AI Assistant

This application is a magical item recommendation assistant built using Generative AI and RAG.

##  Features
- Ask for magical items or potions (e.g., "I need a stealth potion")
- Uses Retrieval-Augmented Generation with a vector DB (FAISS)
- Streamlit frontend with a magical UI

##  Setup Instructions

1. Clone the repo and place your PDF as `The_Enchanted_Emporium.pdf`.
2. Build the vector DB:

python vector_db_builder.py


3. Run the app:


python3 -m streamlit run Rag.py 

## Requirements

Install dependencies with:
streamlit
PyPDF2
langchain
langchain-community
langchain-google-genai
google-api-core
faiss-cpu



Enjoy shopping magically! ✨# Rag-Application