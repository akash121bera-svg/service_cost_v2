# Service Costing RAG Application Workflow Blueprint (Updated)

This document outlines the complete architectural workflows, data processing sequences, and search-enriched costing logic used in the **Service Costing RAG** application, fully updated for the new hybrid search pipeline (DuckDuckGo + Tavily).

It covers both entry points:
1. **Standalone Streamlit Routing (Legacy Standalone Mode)**
2. **API-Driven Hybrid Workflow Intelligence Orchestrator (FastAPI + Staged Pipeline)**

---

## 1. Unified Multi-Entry Architecture Overview

The system operates under a dual-entry structure. Users can interact via a standalone desktop interface or via an enterprise REST API.

```mermaid
flowchart TB
    subgraph UI["User Interfaces (Dual-Entry)"]
        UI_Streamlit["Streamlit Frontend (streamlit_app.py)"]
        UI_API["FastAPI Backend Endpoint (main.py /api/query)"]
    end

    subgraph Direct_Routing["Direct Streamlit Routing (Standalone Mode)"]
        D_Classify{"Classification Rules"}
        D_Web["Direct Web Search (DuckDuckGo default, Tavily fallback)"]
        D_Struct["Deterministic Table Builder (uploaded_costs.py)"]
        D_RAG["Local RAG pipeline (pipeline.py)"]
        D_FAISS[("FAISS Vector DB")]
    end

    subgraph Orchestrator_Loop["Hybrid Workflow Intelligence Orchestrator (engine/orchestrator.py)"]
        O_Classify["Gemini Query Classifier"]
        O_Contract["Execution Contract Generator"]
        
        subgraph Stage_1["Stage 1: Context Gathering (Sequential)"]
            S1_Mem["MEM_LOAD (engine/memory.py)"]
            S1_RAG["RAG_RETRIEVAL (rag/pipeline.py)"]
            S1_Web["WEB_SEARCH (engine/search_enrichment.py)"]
        end
        
        subgraph Stage_2["Stage 2: Core Processing (Thread-Safe Parallel)"]
            direction LR
            S2_Cost["COSTING (engine/uploaded_costs.py)"]
            S2_Comp["COMPLIANCE (engine/compliance.py)"]
        end
        
        S2_Logic["VENDOR_LOGIC (Scoring & Ranking)"]
        
        subgraph Stage_3["Stage 3: Normalization (engine/vendor_normalization.py)"]
            S3_Norm["Vendor Entity Normalization"]
        end
        
        subgraph Stage_4["Stage 4: LLM Synthesis"]
            S4_Synth["Llama 3.2 Synthesis (ChatGroq)"]
        end
    end

    subgraph Infrastructure["Shared Back-End Infrastructure"]
        I_Cache["Enterprise Cache (engine/caching.py)"]
        I_Factors["Formula Modules (factors/)"]
        I_Vision["Vision/OCR Table Extractor (engine/vision_extractor.py)"]
    end

    %% Wiring Streamlit Standalone
    UI_Streamlit --> D_Classify
    D_Classify -->|Web Term| D_Web
    D_Classify -->|Structured Term| D_Struct
    D_Classify -->|Open-ended Term| D_RAG
    D_RAG <--> D_FAISS

    %% Wiring API Entry
    UI_API --> O_Classify
    
    %% Wiring Orchestrator
    O_Classify --> O_Contract
    O_Contract --> S1_Mem
    O_Contract --> S1_RAG
    O_Contract --> S1_Web
    
    S1_RAG --> S2_Cost
    S1_Web --> S2_Cost
    
    S2_Cost & S2_Comp --> S2_Logic
    S2_Logic --> S3_Norm
    S3_Norm --> S4_Synth
    
    %% Wiring Infrastructure connections
    S2_Cost --> I_Factors
    S2_Comp & S2_Cost & S1_Web & S1_RAG <--> I_Cache
    UI_Streamlit & UI_API --> I_Vision
```

---

## 2. The Staged Hybrid Workflow Orchestration (FastAPI Pathway)

When queried via the FastAPI server, the request runs through a **4-stage sequential and parallel execution pipeline** managed by `engine/orchestrator.py`:

```mermaid
flowchart TD
    In([User Question + Quantity]) --> Step1[LLM Classifier]
    Step1 --> Step2[Execution Contract Generator]
    
    subgraph S1["Stage 1: Context Gathering (Sequential)"]
        S1_1[MEM_LOAD: Retrieve Chat Continuity] --> S1_2[RAG_RETRIEVAL: Similarity search FAISS]
        S1_2 --> S1_3[WEB_SEARCH: DuckDuckGo search -> Tavily fallback]
    end
    
    Step2 --> S1
    
    subgraph S2["Stage 2: Core Analysis & Audit (Parallel thread pool)"]
        direction LR
        S2_Cost["COSTING: Deterministic service calculations"]
        S2_Comp["COMPLIANCE: Certifications scan & Trust scoring"]
    end
    
    S1 --> S2
    
    S2 --> S2_Dep[VENDOR_LOGIC: Rank vendors by cost & compliance]
    
    subgraph S3["Stage 3: Normalization & Aggregation"]
        S3_Norm[Normalize local CSV vendor profiles & web discovered leads]
    end
    
    S2_Dep --> S3
    
    subgraph S4["Stage 4: Synthesized Response"]
        S4_Synth[Llama 3.2 StrOutputParser QA Synthesis]
    end
    
    S3 --> S4
    S4 --> Out([Unified Markdown Answer + State JSON])
```

### Breakdown of the 4 Stages

#### Stage 1: Context Gathering (Sequential)
- **MEM_LOAD** (`engine/memory.py`): Restores the conversation continuity rules.
- **RAG_RETRIEVAL** (`rag/pipeline.py`): Executes similarity searches on vector embeddings of uploaded PDFs/CSVs.
- **WEB_SEARCH** (`engine/search_enrichment.py`): Performs a hybrid search:
  * **Default (100% Free)**: Queries **DuckDuckGo Search** (using `duckduckgo-search` library). It targets verified vendor portals (`indiamart.com`, `tradeindia.com`, `alibaba.com`, etc.) without requiring any API keys.
  * **Fallback**: If DuckDuckGo gets rate-limited (empty results) or raises an exception, the pipeline instantly and silently falls back to **Tavily Search** (if `TAVILY_API_KEY` is configured).

#### Stage 2: Core Analysis & Auditing (Concurrent Thread Pool)
Runs parallel workloads via `ThreadPoolExecutor` to minimize processing latency:
* **COSTING** (`engine/uploaded_costs.py`): Calculates the total service cost using shipment quantity and selected costing factors.
* **COMPLIANCE** (`engine/compliance.py`): Audits extracted texts and search hits using regex validation rules.

##### Compliance Trust Scoring Model
Trust scores range between $0.0$ and $1.0$ based on verified certifications:
* **ISO 13485**: `0.35` (Medical Quality)
* **FDA Approved/Registered**: `0.30` (Federal Safety)
* **CE Mark**: `0.20` (European Compliance)
* **GMP**: `0.10` (Good Manufacturing)
* **ISO 9001**: `0.05` (General Quality)

##### Deterministic Cost Formula
$$\text{Total Cost} = \sum (\text{Packaging Rate} + \text{Sterilization Rate} + \text{Logistics Rate} + \text{Quality Rate} + \text{Warehousing Rate}) \times \text{Quantity}$$

* **VENDOR_LOGIC**: Executes immediately after costing is complete to rank vendors.

#### Stage 3: Normalization & Aggregation
Uses `engine/vendor_normalization.py` to organize structured data fields (costing breakdowns, contact details, verified certifications, risk flags, and lead times) for both CSV-based rate-card vendors and external web leads.

#### Stage 4: Synthesis & Output
Generates a structured report using Llama 3.2 via ChatGroq, providing tables, risk flags, and an objective recommendation.

---

## 3. Standalone Streamlit Query Flow (Fallback Routing)

If the Streamlit application is run locally without the API backend, it relies on pattern-matching routing rules:

```mermaid
flowchart TD
    Start([User Question]) --> Q_Val{Is Service Cost<br/>Question?}
    Q_Val -->|No| Reject[Explain domain limits]
    
    Q_Val -->|Yes| Web_Decide{Requires Web Search?}
    
    Web_Decide -->|Yes| Web_Check{Has location & category?}
    Web_Check -->|No| Prompt_Info[Prompt user for location/category]
    Web_Check -->|Yes| DDG_Search[Execute DuckDuckGo Search]
    DDG_Search -->|Failed / Rate-Limited| Tavily_Search[Fallback: Execute Tavily Search]
    
    DDG_Search --> Filter_Results[Filter out electricity, power, construction]
    Tavily_Search --> Filter_Results
    
    Filter_Results --> Ext_Contact[Extract phone numbers]
    Filter_Results --> Calc_Reliability[Calculate reliability score]
    Filter_Results --> Present_Web[Display results list & map]
    
    Web_Decide -->|No| Struct_Decide{Matches Structured CSV terms?}
    
    Struct_Decide -->|Yes| Deterministic_Route[Call uploaded_costs.py]
    Deterministic_Route --> Match_Vendor[Find vendor matches by name/id]
    Match_Vendor --> Match_Qty[Find shipment category band]
    Match_Qty --> Cal_Breakdown[Generate costing factor table]
    Cal_Breakdown --> Present_Table[Display interactive data table]
    
    Struct_Decide -->|No| RAG_Decide{RAG Chunks Available?}
    RAG_Decide -->|No| Ask_Upload[Ask user to upload CSV/PDF]
    RAG_Decide -->|Yes| FAISS_Retrieve[Retrieve nearest chunks]
    FAISS_Retrieve --> Gemini_RAG[Gemini RAG Pipeline generation]
    Gemini_RAG --> Present_Text[Display text answer]
```

---

## 4. Ingestion & Preprocessing Workflow

Processes raw uploads into vector index chunks and dataframes:

```mermaid
flowchart LR
    subgraph Input["Uploads"]
        I1[CSV File]
        I2[PDF File]
        I3[PNG/JPG Image]
    end

    subgraph Processing["Parsers & Extractors"]
        P1[CSV Parser<br/>pandas]
        P2[PDF Extractor<br/>PyMuPDF]
        P3[OCR Table Extractor<br/>vision_extractor.py]
    end

    subgraph Outputs["Ingestion Results"]
        O1[Local CSV DataFrame]
        O2[Text Chunks<br/>900 chars]
        O3[FAISS Vector Store<br/>Sentence Transformers]
    end

    I1 --> P1
    I2 --> P2
    I3 --> P3
    
    P1 --> O1
    P1 --> O2
    P2 --> O2
    P3 --> O1
    
    O2 --> O3
```

---

## 5. Shared Enterprise Caching Layer

To optimize performance and minimize external API expenses, all costing calculations, similarity checks, web searches, and compliance audits utilize a thread-safe caching system in `engine/caching.py`:

```text
Function Parameters ---> Argument Serializer ---> MD5 Hash Generator ---> Cache Lookup (Get/Set)
```

- **Thread-Safety**: Governed by reentrant locks (`threading.Lock`).
- **Keys**: Arguments and sorted keyword parameters are serialized to produce standard MD5 hash identifiers.
