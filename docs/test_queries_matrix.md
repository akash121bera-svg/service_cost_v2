# Service Costing RAG Application - Exhaustive Testing Matrix

This testing matrix provides an exhaustive catalog of test queries, divided by category and operational pathway, to validate all features and edge cases of the Service Costing RAG application.

---

## Category 1: Deterministic Costing Engine (Local CSV Pathway)

These queries test mathematical accuracy, pricing factor calculations, multi-vendor comparisons, shipment quantity scales, and specific rate retrievals from local dataframes.

| Query ID | Test Query | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q1.1** | What is the total service cost for 250 units from Biosafe Packaging? | Single Costing | Deterministic rate cards applied for 250 quantity band across all 5 cost factors. |
| **Q1.2** | Calculate the overall pricing for 850 units from Carechain Services. | Band Boundary Test | Should parse band for high volume (e.g. 500-1000 band) and apply the correct rates. |
| **Q1.3** | Compare all vendors for a quantity of 500 units and recommend the cheapest. | Comparison Engine | Computes total costs across all CSV dataframes, highlights cheapest, lists others in ascending order. |
| **Q1.4** | Show the packaging and warehousing rates for all vendors at 100 units. | Specific Factors | Interactive table showing columns for `packaging_rate` and `warehousing_rate` across all vendors. |
| **Q1.5** | What are the sterilization and quality inspection rates for Carechain Services at 1000 units? | Specific Factors | Highlights sterilization and quality inspection specific rates for high-volume shipment bands. |
| **Q1.6** | Compare logistical overhead costs between Biosafe Packaging and Carechain Services for 300 units. | Multi-Vendor Comparative | Generates comparative breakdown specifically for the `logistics_rate` factor at 300 units. |
| **Q1.7** | What is the total cost breakdown for 50 units from Medpack Solutions? | Low-Volume Band | Tests minimum billing/low-volume band rates for all factors. |
| **Q1.8** | Recommend the most cost-effective warehousing supplier for a medium shipment size. | Semantic Volume | Maps "medium shipment size" to quantity band and ranks warehousing costs. |
| **Q1.9** | Show the complete comparative factor analysis for 1200 units across all suppliers. | Scale/Max Band | Computes total costs for the maximum scale band across all rate cards. |
| **Q1.10**| Rank all active suppliers based on their quality audit fees at 400 quantity. | Factor Ranking | Ranks all vendors by their `quality_rate` factor. |

---

## Category 2: Unstructured Document RAG (FAISS Pathway)

These queries verify text parsing (PDF/CSV), similarity search indexing, and LLM text generation capabilities for non-deterministic insights.

| Query ID | Test Query | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q2.1** | What compliance certificates or regulatory histories are mentioned in the files? | Document Context | Retrieves matching PDF page chunks mentioning ISO, FDA, or compliance certificates. |
| **Q2.2** | Are there any capacity limitations or minimum order quantities noted in the rate cards? | Edge constraints | Similarity search pulls chunks relating to capacities, maximum limits, or MOQ. |
| **Q2.3** | What are the warehouse rental contract terms and insurance policies described in the document? | Document Context | Locates insurance clauses, liability coverage values, or warehouse rental policies. |
| **Q2.4** | Summarize the shipment handling and transportation guidelines from the uploaded files. | Synthesis | Summarizes text chunks dealing with logistics, transport safety, and fragile handling. |
| **Q2.5** | List any special packaging specifications, pouch dimensions, or carton constraints. | Detailed Spec | Pulls specifications, dimensions, temperature conditions, or pouch size limits. |
| **Q2.6** | What is the escalation process or penalty clause in case of a vendor shipment delay? | SLA Context | Retrieves text relating to service level agreements (SLAs), penalties, and dispute resolutions. |
| **Q2.7** | How frequently are quality control audits performed according to the uploaded SOPs? | Audit Context | Locates quality assurance SOP chunks detailing audit frequencies and standards. |

---

## Category 3: Live Supplier Search (Tavily/DDG Priority Pathway)

These queries validate pattern-matching classification, location hint parsing, B2B trusted domain priority filtering, and fallback triggers.

| Query ID | Test Query | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q3.1** | Find medical packaging vendors in Mumbai and list their contact details. | B2B Priority Search | DDG site-filtered search targeting `indiamart.com`, `tradeindia.com` in Mumbai. |
| **Q3.2** | Who are the top 5 sterilization service providers in Delhi NCR? | Location Parse | Extracts "Delhi NCR" and searches for sterilization providers near Delhi. |
| **Q3.3** | List ISO certified cleanroom warehousing suppliers near Chennai. | Certificate + Location | Queries ISO cleanroom warehouse suppliers near Chennai. |
| **Q3.4** | Search the web for medical grade logistics companies in Bangalore with contacts. | Search Fallback | Falls back to Tavily if DDG rate-limits; extracts phone numbers. |
| **Q3.5** | Find packaging suppliers near Ahmedabad. | Location Parse | Extracts Ahmedabad and returns localized packaging manufacturers. |
| **Q3.6** | List active logistics distributors located in Kolkata. | Location Parse | Queries logistics distributors in Kolkata. |
| **Q3.7** | Find sterilization facilities in Pune. | Location Parse | Locates sterilization companies in Pune. |
| **Q3.8** | Who are the trusted medical warehousing suppliers in Hyderabad? | Location Parse | Queries medical grade warehousing suppliers in Hyderabad. |
| **Q3.9** | List medical device inspection and quality audit vendors near Gurgaon. | Location Parse | Finds QA/QC audit service providers near Gurgaon. |
| **Q3.10**| Find pouch packaging and labeling companies in Mumbai with PIN 400001. | PIN Extraction | Parses postal PIN "400001" and restricts searches specifically around that code. |

---

## Category 4: Parallel Web Scraping & Dynamic Contacts (Scraper Pathway)

These queries evaluate parallel thread execution, HTML parsing robustness, timeout handling, and regex phone/contact extractions from page bodies.

| Query ID | Test Query | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q4.1** | Scrape the official website contacts for sterilization providers near Pune. | Scraper + Scraping | CrawlsPune sterilization website URLs in parallel, extracting direct body content. |
| **Q4.2** | Find logistics partners in Chennai and extract phone numbers from their websites. | Phone Extraction | Performs search, crawls links, and runs regex patterns on HTML text to extract phone contacts. |
| **Q4.3** | Search B2B portals for medical warehouse suppliers and pull their direct contact details. | Scraper + B2B | Concurrently crawls Indiamart/Alibaba pages and extracts B2B listings' numbers. |

---

## Category 5: Regulatory Compliance & Trust Scoring (Audit Pathway)

These queries verify regex validation rule mappings and compliance weighting trust models (ISO 13485, CE, FDA, GMP, ISO 9001).

| Query ID | Test Query | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q5.1** | Which of the discovered Mumbai packaging suppliers are FDA registered? | Compliance Scan | Scrape matches; runs `fda` pattern validations and logs trust weights. |
| **Q5.2** | Check ISO 13485 and CE compliance scores for logistics vendors in Chennai. | Compliance Scan | Scans search snippets for `iso 13485` and `ce`; calculates compliance trust scores. |
| **Q5.3** | Compare trust scores between local vendors based on their GMP and ISO 9001 audit logs. | Compliance Scan | Scans rate card data for certifications; weighs GMP (`0.10`) vs ISO 9001 (`0.05`). |

---

## Category 6: Multi-Turn Conversation Memory (Continuity Pathway)

These queries validate the lightweight memory layer, ensuring conversational continuity rules and vendor-switch context are properly preserved.

| Query ID | Test Query (Turn 1 $\rightarrow$ Turn 2) | Target Component | Expected Routing & Output |
| :--- | :--- | :--- | :--- |
| **Q6.1** | **Turn 1**: What is the cost for 200 units from Biosafe?<br>**Turn 2**: What if the quantity changes to 500 units? | Contextual Memory | Turn 2 rephrases to: "What is the cost for 500 units from Biosafe?" and preserves the vendor. |
| **Q6.2** | **Turn 1**: List sterilization vendors near Delhi.<br>**Turn 2**: Show their ISO certifications. | Contextual Memory | Turn 2 rephrases to: "Show the ISO certifications for sterilization vendors near Delhi." |
| **Q6.3** | **Turn 1**: Compare Biosafe and Carechain at 100 units.<br>**Turn 2**: Who is best suited for 1000 units instead? | Switched Volume | Turn 2 rephrases to: "Compare Biosafe and Carechain at 1000 units and recommend the cheapest." |
