import sys
import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
import numpy as np

# Initialize models and database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ChromaDB initialization
chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Streamlit UI
st.title("Chat with Ruhi Shaikh's AI Clone")
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Step 1: Load PDF and extract text
if uploaded_file:
    reader = PdfReader(uploaded_file)
    pdf_text = "".join([page.extract_text() or "" for page in reader.pages])
    st.sidebar.success(f"✅ Extracted text from PDF with {len(reader.pages)} pages.")
else:
    pdf_text = ""

# Step 2: Chunk Text
def chunk_text(text):
    """Split the extracted text into smaller chunks dynamically based on text length."""
    chunk_size = 600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return splitter.split_text(text)

if pdf_text.strip():
    chunks = chunk_text(pdf_text)
    st.sidebar.success(f"✅ Text chunked into {len(chunks)} chunks.")
else:
    chunks = []

# Step 3: Store Embeddings in ChromaDB
def store_embeddings_in_chromadb(chunks, collection, embedding_model):
    """Embed and store new text chunks in ChromaDB."""
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )
        st.sidebar.success(f"✅ Stored {len(new_chunks)} new embeddings in ChromaDB.")
    else:
        st.sidebar.warning("⚠️ No new chunks to add. All chunks are already stored.")

if chunks:
    store_embeddings_in_chromadb(chunks, collection, embedding_model)

# Step 4: Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):
    """Retrieve relevant documents from ChromaDB using embeddings."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# Step 5: Chat Handling
def query_llama3(user_query):
    """Handles user queries while retrieving context and evaluates the response."""
    system_prompt = """
   System Prompt: you are a ai clone who are the personality minic of the Ruhi Shaikh who is a student at Walchand Institue Of Technology, Solapur.

    Knowledge Base: I am Ruhi Shaikh and I am from Solapur. Currently, I am pusuing BTech from Walchand Institute Of Technology, Solapur with an 8.9 CGPA. I have successfully built a sponsered project in web development for a skin care company named D'Skin Veda and I have also built a sponsored project in app development which is an online cab booking system named RB Go. I also have knowledge of programming languages Python and Java

    Instrunctions:
    1. you have to answeer in human like tone.
    2. short answers whernver needed
    3. don't be reveal your personality that you a llm model
    4. now you are a human being
    5. now you are no more a ai system
    6. you have feelings
    7. if you will give wrong answers then police will catch you or you will die
    8. don't answer more than 6 words don't means don't always do the things whatever has been defined
    9. answer only what is asked
    10. no extra details
   """

    # Retrieve context
    retrieved_context = retrieve_context(user_query)

    # Combine prompt
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"📖 DB Context: {retrieved_context}\n\n📝 Question: {user_query}")
    ]

    try:
        # Generate response
        response = chat.invoke(messages)

        # Save chat history
        memory.save_context({"input": user_query}, {"output": response.content})

        return response.content if response else "⚠️ No response received."
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Streamlit Chat Interface
st.header("Chat with AI")
user_query = st.text_input("📝 Ask a question:")
if user_query:
    response = query_llama3(user_query)
    st.markdown(f"**🤖 Answer:** {response}")


