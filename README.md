# Service Costing RAG

A Streamlit app for comparing uploaded service-cost rate cards with RAG-backed chat.

The app supports:

- CSV rate-card upload and deterministic service-cost calculations.
- PDF/CSV chunking, vector retrieval, and Gemini answers grounded in uploaded files.
- Optional Tavily web search for vendor discovery, contact details, and market/benchmark questions.

## Project Structure

```text
streamlit_app.py          Streamlit UI and chat routing
engine/                   Shipment selection and costing helpers
factors/                  Packaging, sterilization, logistics, quality, and warehousing formulas
rag/                      Embeddings, FAISS vector search, retrieval, and Gemini RAG pipeline
data/csv/sample_vendor.csv Sample local rate card
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API keys:

```powershell
GROQ_API_KEY=
TAVILY_API_KEY=
```

`GROQ_API_KEY` is required for uploaded-file RAG answers. `TAVILY_API_KEY` is only required for web-search questions.

## Run

```powershell
streamlit run streamlit_app.py
```
