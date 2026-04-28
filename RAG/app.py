import os
import uuid
import numpy as np
import faiss
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embedding model once at startup
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_tokens=512)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask).sum(1) / input_mask.sum(1).clamp(min=1e-9)


def get_embeddings(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    embeddings = mean_pooling(output, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy().astype("float32")


def process_pdf(filepath):
    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


# In-memory store: session_id -> {index, texts}
stores = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    session_id = uuid.uuid4().hex[:8]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    file.save(filepath)

    chunks = process_pdf(filepath)
    os.remove(filepath)

    texts = [c.page_content for c in chunks]
    embeddings = get_embeddings(texts)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine on normalized vecs
    index.add(embeddings)

    stores[session_id] = {"index": index, "texts": texts}
    return jsonify({"session_id": session_id, "chunks": len(texts)})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    session_id = data.get("session_id")
    question = data.get("query", "").strip()

    if not session_id or session_id not in stores:
        return jsonify({"error": "Upload a PDF first"}), 400
    if not question:
        return jsonify({"error": "Query cannot be empty"}), 400

    store = stores[session_id]
    q_emb = get_embeddings([question])
    _, indices = store["index"].search(q_emb, k=3)

    context = "\n\n".join(store["texts"][i] for i in indices[0] if i < len(store["texts"]))
    if not context:
        return jsonify({"answer": "No relevant context found."})

    prompt = f"""Use the given context to answer the question concisely.

Context:
{context}

Question:
{question}"""

    response = llm.invoke([prompt])
    return jsonify({"answer": response.content})


if __name__ == "__main__":
    app.run(debug=True)
