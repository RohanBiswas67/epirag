"""
EpiRAG - server.py
------------------
Flask server with:
  - /api/query        — standard JSON response
  - /api/query/stream — SSE streaming (live debate events)
  - /api/stats        — corpus stats
  - /api/metrics      — session performance metrics
  - /performance      — performance dashboard page
"""

import os
import time
import json
import queue
import threading
import chromadb
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from query import rag_query, set_components

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

COLLECTION_NAME = "epirag"
IS_CLOUD        = os.environ.get("EPIRAG_ENV", "").lower() == "cloud"

# -- Session metrics ------------------------------------------------------------
SESSION_METRICS = {
    "queries_total":    0,
    "queries_local":    0,
    "queries_web":      0,
    "queries_hybrid":   0,
    "queries_debate":   0,
    "latencies_ms":     [],
    "started_at":       time.time(),
}

def record_metric(result, elapsed_ms):
    SESSION_METRICS["queries_total"] += 1
    SESSION_METRICS["latencies_ms"].append(elapsed_ms)
    mode = result.get("mode", "")
    if mode == "local":   SESSION_METRICS["queries_local"]  += 1
    if mode == "web":     SESSION_METRICS["queries_web"]    += 1
    if mode == "hybrid":  SESSION_METRICS["queries_hybrid"] += 1
    if result.get("is_debate"): SESSION_METRICS["queries_debate"] += 1


# -- Corpus startup------------------------------------------------------------
_collection  = None
_embedder    = None
CORPUS_STATS = {}

def init_corpus():
    global _collection, _embedder, CORPUS_STATS
    if IS_CLOUD:
        print("Cloud mode - building in-memory corpus", flush=True)
        from ingest import build_collection_in_memory
        _collection, _embedder = build_collection_in_memory()
    else:
        print("Local mode - loading from ./chroma_db/", flush=True)
        from sentence_transformers import SentenceTransformer
        client      = chromadb.PersistentClient(path="./chroma_db")
        _collection = client.get_collection(COLLECTION_NAME)
        _embedder   = SentenceTransformer("all-MiniLM-L6-v2")

    set_components(_embedder, _collection)

    count   = _collection.count()
    results = _collection.get(limit=count, include=["metadatas"])
    papers  = sorted(set(
        m.get("paper_name", m.get("source", "Unknown"))
        for m in results["metadatas"]
    ))
    CORPUS_STATS.update({
        "chunks":    count,
        "papers":    len(papers),
        "paperList": papers,
        "status":    "online",
        "mode":      "cloud (in-memory)" if IS_CLOUD else "local (persistent)"
    })
    print(f"Corpus ready: {count} chunks / {len(papers)} papers", flush=True)


init_corpus()


# -- Routes ------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/performance")
def performance():
    return send_from_directory("static", "performance.html")


@app.route("/api/stats")
def stats():
    return jsonify(CORPUS_STATS)


@app.route("/api/metrics")
def metrics():
    lats = SESSION_METRICS["latencies_ms"]
    avg  = int(sum(lats) / len(lats)) if lats else 0
    return jsonify({
        **SESSION_METRICS,
        "avg_latency_ms": avg,
        "uptime_seconds": int(time.time() - SESSION_METRICS["started_at"]),
        "latencies_ms":   lats[-50:],   # last 50 only
    })


@app.route("/api/query", methods=["POST"])
def query():
    data     = request.json or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    groq_key   = os.environ.get("GROQ_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    hf_token   = os.environ.get("HF_TOKEN")
    if not groq_key:
        return jsonify({"error": "GROQ_API_KEY not set on server"}), 500

    start  = time.time()
    result = rag_query(
        question,
        groq_api_key   = groq_key,
        tavily_api_key = tavily_key,
        hf_token       = hf_token,
        use_debate     = bool(hf_token)
    )
    elapsed_ms = int((time.time() - start) * 1000)
    record_metric(result, elapsed_ms)

    return jsonify({
        "answer":        result["answer"],
        "sources":       result["sources"],
        "mode":          result["mode"],
        "avg_sim":       result["avg_sim"],
        "latency_ms":    elapsed_ms,
        "tokens":        len(result["answer"]) // 4,
        "question":      question,
        "is_debate":     result.get("is_debate", False),
        "debate_rounds": result.get("debate_rounds", []),
        "consensus":     result.get("consensus", False),
        "rounds_run":    result.get("rounds_run", 0),
    })


@app.route("/api/query/stream", methods=["POST"])
def query_stream():
    """
    SSE endpoint. Streams debate events in real time, then sends final result.

    Event types sent to browser:
      data: {"type": "status",      "text": "..."}
      data: {"type": "round_start", "round": N}
      data: {"type": "agent_done",  "round": N, "name": "...", "color": "...", "text": "..."}
      data: {"type": "synthesizing"}
      data: {"type": "result",      ...full result payload...}
      data: {"type": "error",       "text": "..."}
    """
    data     = request.json or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    groq_key   = os.environ.get("GROQ_API_KEY")
    tavily_key = os.environ.get("TAVILY_API_KEY")
    hf_token   = os.environ.get("HF_TOKEN")

    event_queue = queue.Queue()

    def callback(event):
        event_queue.put(event)

    def run_in_thread():
        try:
            start  = time.time()
            result = rag_query(
                question,
                groq_api_key   = groq_key,
                tavily_api_key = tavily_key,
                hf_token       = hf_token,
                use_debate     = bool(hf_token),
                sse_callback   = callback
            )
            elapsed_ms = int((time.time() - start) * 1000)
            record_metric(result, elapsed_ms)
            event_queue.put({
                "type":        "result",
                "answer":      result["answer"],
                "sources":     result["sources"],
                "mode":        result["mode"],
                "avg_sim":     result["avg_sim"],
                "latency_ms":  elapsed_ms,
                "tokens":      len(result["answer"]) // 4,
                "is_debate":   result.get("is_debate", False),
                "debate_rounds": result.get("debate_rounds", []),
                "consensus":   result.get("consensus", False),
                "rounds_run":  result.get("rounds_run", 0),
            })
        except Exception as e:
            event_queue.put({"type": "error", "text": str(e)})
        finally:
            event_queue.put(None)  # sentinel

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    def generate():
        yield "data: " + json.dumps({"type": "status", "text": "Retrieving context..."}) + "\n\n"
        while True:
            try:
                event = event_queue.get(timeout=60)
            except queue.Empty:
                yield "data: " + json.dumps({"type": "error", "text": "Timeout"}) + "\n\n"
                break
            if event is None:
                break
            yield "data: " + json.dumps(event) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "corpus": CORPUS_STATS.get("status", "unknown")})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)