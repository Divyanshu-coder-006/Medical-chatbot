from flask import Flask, render_template, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

# Load API keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Globals for lazy loading
embeddings = None
docsearch = None
rag_chain = None

def initialize_rag_chain():
    global embeddings, docsearch, rag_chain
    if rag_chain is None:
        print("Initializing embeddings and Pinecone index...")
        embeddings = download_embeddings()
        print("Embeddings downloaded.")

        index_name = "medical-chatbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        print("Pinecone index loaded.")

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        chatModel = ChatOpenAI(model="gpt-4o")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAG chain ready.")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "No message provided", 400

    # Lazy initialize RAG chain
    initialize_rag_chain()

    print("Input:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return response["answer"]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



