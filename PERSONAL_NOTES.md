# Personal Project Notes

This file is for understanding the repo quickly.

## Big Picture

`streamlit_app.py` is the app entrypoint. It handles the UI, upload flow, chat display, and routing between three answer paths:

- Structured CSV answers for exact local questions such as available vendors, rate lookup, best option, and comparison factors.
- Uploaded-file RAG for open-ended questions about uploaded CSV/PDF content.
- Tavily web search for live vendor discovery, contacts, market rates, benchmarks, and location-based supplier questions.

## Directory Map

```text
engine/       Deterministic service-cost logic and CSV chat helpers.
factors/      Small formula modules for each cost factor.
rag/          Uploaded-file chunking, embeddings, FAISS retrieval, and Gemini answer generation.
data/         Local sample data only.
```

## Engine

- `category_selector.py` picks the shipment quantity band from an uploaded rate card.
- `costing_engine.py` calculates total service cost from either simple rate columns or detailed factor columns.
- `uploaded_costs.py` prepares CSV-backed answers for the chat, such as vendor lists, rate lookups, best-option comparisons, and comparison-factor tables.

Keep Streamlit, Gemini, and Tavily code out of `engine/`.

## Factors

Each subdirectory contains one service-cost formula:

- `packaging/`: pouch, label, and carton cost.
- `sterilization/`: batch sterilization and validation spread across units per batch.
- `logistics/`: transport, handling, and distributor fees spread across quantity.
- `quality/`: inspection, audit, and documentation spread across quantity.
- `warehousing/`: rent, insurance, and inventory handling spread across quantity.

These functions stay small so `engine/costing_engine.py` is easy to audit.

## RAG

- `embedding.py` creates local sentence-transformer embeddings.
- `vector_store.py` builds a FAISS index.
- `retriever.py` returns nearest chunks for a user question.
- `pipeline.py` converts uploaded CSV/PDF files into chunks, retrieves context, and asks Gemini to answer from that context.

Tavily web search is not in `rag/`; it is routed from `streamlit_app.py` because it answers live web/vendor discovery questions.

## Data

- `data/csv/sample_vendor.csv` is a detailed rate-card example with factor-level columns.

Uploaded files from Streamlit are handled in memory and are not saved to `data/`.

## Simple CSV Schema

The app also supports simple uploaded rate cards with:

- `shipment_category`
- `min_qty`
- `max_qty`
- `packaging_rate`
- `sterilization_rate`
- `logistics_rate`
- `quality_rate`
- `warehousing_rate`
