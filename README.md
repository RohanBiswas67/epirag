# EpiRAG

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Spaces-yellow?logo=huggingface)](https://rohanb67-epirag.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-RohanBiswas67-181717?logo=github)](https://github.com/RohanBiswas67/epirag)
[![Dataset](https://img.shields.io/badge/Dataset-RohanB67%2Fpapers-blue?logo=huggingface)](https://huggingface.co/datasets/RohanB67/papers)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-rohan--biswas--0rb-0A66C2?logo=linkedin)](https://linkedin.com/in/rohan-biswas-0rb)

A hybrid agentic RAG system for querying epidemic modeling and network science literature.

Built because ctrl+F across 20 PDFs is not a research workflow. Ask it a question, get a cited answer from the actual papers. If the corpus does not have it, it falls back to live web search automatically.

---

## Architecture
 
```
HF Dataset (RohanB67/papers)
    PyMuPDF -- text extraction + chunking (500 chars, 100 overlap)
    sentence-transformers -- all-MiniLM-L6-v2 embeddings
    ChromaDB EphemeralClient -- in-memory vector store
    Confidence router (sim threshold 0.45 + recency keywords)
    Local corpus OR DuckDuckGo / Tavily web search
    Multi-agent debate swarm (4 models argue, Epsilon synthesizes)
    Flask + SSE -- real-time debate streaming to browser
```
 
### Multi-agent debate
 
Five agents with different models and personalities debate each query:
 
| Agent | Model | Provider | Role |
|---|---|---|---|
| Alpha | Llama 3.3 70B Versatile | Groq | Skeptic |
| Beta | GPT-OSS 120B | Groq | Deep Reasoner |
| Gamma | Llama 4 Scout 17B 16E | Groq | Connector |
| Delta | Qwen3-32B | Groq | Literalist |
| Epsilon | GPT-OSS 20B | Groq | Synthesizer |
 
Agents run in parallel. Up to 3 rounds of debate with convergence detection. Epsilon synthesizes the final answer from the full transcript.
 
### Retrieval logic
 
 
| Condition | Behaviour |
|---|---|
| Local similarity >= 0.45 | Answered from corpus only |
| Local similarity < 0.45 | DuckDuckGo triggered, Tavily as fallback |
| Query contains "latest / recent / 2025 / 2026 / new / today" | Web search forced regardless of score |

### Citation enrichment
 
For every local source, the system queries Semantic Scholar, OpenAlex, and PubMed E-utils to surface DOI, arXiv ID, open-access PDF, and PubMed links. Falls back to generated search links (Google Scholar, NCBI, arXiv) when exact matches are not found.
 
---
 
## Corpus
 
19 papers across epidemic modeling, network science, causal inference, and graph theory. Includes Shalizi & Thomas (2011), Myers & Leskovec (2010), Britton, Guzman, Groendyke, Netrapalli, Clauset, Handcock & Jones, Spirtes Glymour Scheines, and others. All related to my independent research on observational equivalence classes and non-identifiability in SIS epidemic dynamics on contact networks.
 
Papers are stored in the [RohanB67/papers](https://huggingface.co/datasets/RohanB67/papers) HF Dataset and downloaded at startup. No PDFs committed to the repo.
 
---
 
## Stack
 
| Layer | Tool |
|---|---|
| PDF ingestion | PyMuPDF (4-strategy title extraction) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | ChromaDB (ephemeral on cloud, persistent locally) |
| Web search | DuckDuckGo (free) with Tavily fallback |
| Debate LLMs | Groq (Llama 3.3 70B / GPT-OSS 120B / Llama 4 Scout / Qwen3-32B) |
| Synthesis LLM | Groq / GPT-OSS 20B |
| Server | Flask + SSE streaming |
| Deployment | HF Spaces (Docker) |
 
---
 
## Run locally
 
```bash
git clone https://huggingface.co/spaces/RohanB67/epirag
cd epirag
pip install -r requirements.txt
 
cp .env.example .env
# fill in GROQ_API_KEY, TAVILY_API_KEY, HF_TOKEN
 
# put your PDFs in ./papers/ and ingest
python ingest.py
 
# run
python server.py
# open http://localhost:7860
```
 
Set `EPIRAG_ENV=local` to load from local `chroma_db/` instead of downloading from HF Dataset at startup.
 
---
 
## Environment variables
 
| Variable | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | Yes | console.groq.com |
| `HF_TOKEN` | Yes | hf.co/settings/tokens -- enables multi-agent debate |
| `TAVILY_API_KEY` | Optional | app.tavily.com -- web search fallback (1000/month free) |
| `EPIRAG_ENV` | Auto-set | Set to `cloud` by Dockerfile |
 
---
 
Rohan Biswas -- CS grad, IISc FAST-SF research fellow, working on network non-identifiability in epidemic dynamics on contact networks.
