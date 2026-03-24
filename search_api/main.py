from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
import json
import asyncio
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# --- Configurations & DB Setup ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "agriai")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "questions")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

GOLDEN_MONGO_URI = os.getenv("GOLDEN_MONGO_URI") or MONGO_URI
GOLDEN_DB_NAME = os.getenv("GOLDEN_DB_NAME", "golden_db")
GOLDEN_QA_COLLECTION = os.getenv("GOLDEN_QA_COLLECTION", "agri_qa_latest")
GOLDEN_POP_COLLECTION = os.getenv("GOLDEN_POP_COLLECTION", "pop")
GOLDEN_VECTOR_INDEX_NAME = os.getenv("GOLDEN_VECTOR_INDEX_NAME", "vector_index")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en")
MAPPINGS_FILE = "mappings.json"

reviewer_client = MongoClient(MONGO_URI)
reviewer_collection = reviewer_client[DB_NAME][COLLECTION_NAME]

golden_client = MongoClient(GOLDEN_MONGO_URI)
golden_qa_collection = golden_client[GOLDEN_DB_NAME][GOLDEN_QA_COLLECTION]
golden_pop_collection = golden_client[GOLDEN_DB_NAME][GOLDEN_POP_COLLECTION]

print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}...", flush=True)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model loaded successfully.", flush=True)

MAPPINGS = {"crops": {}, "states": {}, "districts": {}, "domains": {}, "seasons": {}}

# --- Background Task Logic ---

def update_mappings_from_db():
    print("Fetching distinct filter values from database...", flush=True)
    
    reviewer_crops = reviewer_collection.distinct("details.crop")
    reviewer_states = reviewer_collection.distinct("details.state")
    reviewer_districts = reviewer_collection.distinct("details.district")
    reviewer_domains = reviewer_collection.distinct("details.domain")
    reviewer_seasons = reviewer_collection.distinct("details.season")

    golden_crops = golden_qa_collection.distinct("metadata.Crop")
    golden_states = golden_qa_collection.distinct("metadata.State")
    golden_districts = golden_qa_collection.distinct("metadata.District")
    golden_seasons = golden_qa_collection.distinct("metadata.Season")

    all_crops = [c for c in (reviewer_crops + golden_crops) if c and isinstance(c, str)]
    all_states = [s for s in (reviewer_states + golden_states) if s and isinstance(s, str)]
    all_districts = [d for d in (reviewer_districts + golden_districts) if d and isinstance(d, str)]
    all_domains = [d for d in reviewer_domains if d and isinstance(d, str)]
    all_seasons = [s for s in (reviewer_seasons + golden_seasons) if s and isinstance(s, str)]

    def build_map(items):
        mapping = {}
        for item in items:
            cleaned = "".join(item.split()).lower()
            if cleaned not in mapping:
                mapping[cleaned] = []
            if item not in mapping[cleaned]:
                mapping[cleaned].append(item)
        return mapping

    new_mappings = {
        "crops": build_map(all_crops),
        "states": build_map(all_states),
        "districts": build_map(all_districts),
        "domains": build_map(all_domains),
        "seasons": build_map(all_seasons)
    }

    with open(MAPPINGS_FILE, "w") as f:
        json.dump(new_mappings, f, indent=4)
        
    print("Mappings updated successfully.", flush=True)
    return new_mappings

def load_mappings_from_file():
    if os.path.exists(MAPPINGS_FILE):
        try:
            with open(MAPPINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {MAPPINGS_FILE}: {e}")
    return {"crops": {}, "states": {}, "districts": {}, "domains": {}, "seasons": {}}

async def daily_update_task():
    while True:
        try:
            global MAPPINGS
            MAPPINGS = await asyncio.to_thread(update_mappings_from_db)
        except Exception as e:
            print(f"Failed to update mappings from DB: {e}")
        await asyncio.sleep(86400)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting FastAPI app initialization...", flush=True)
    global MAPPINGS
    MAPPINGS = load_mappings_from_file()
    print(f"Initial mappings loaded from file. Counts: { {k: len(v) for k, v in MAPPINGS.items()} }", flush=True)
    task = asyncio.create_task(daily_update_task())
    yield
    print("Shutting down FastAPI app...", flush=True)
    task.cancel()

app = FastAPI(title="AgriAI Q&A Semantic Search", lifespan=lifespan)

# --- Pydantic Models ---
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    threshold: float = 0.8
    crop: str | None = None
    state: str | None = None
    district: str | None = None
    domain: str | None = None
    season: str | None = None

class QAItem(BaseModel):
    id: str
    question: str
    text: str | None = None
    answer: str | None = None
    details: dict
    status: str
    source: str
    score: float

class GoldenQAItem(BaseModel):
    text: str
    question: str
    answer: str
    metadata: dict
    score: float

class PopItem(BaseModel):
    text: str
    metadata: dict
    score: float

class MultiSearchResponse(BaseModel):
    reviewer: list[QAItem]
    golden: list[GoldenQAItem]
    pop: list[PopItem]

# --- Helper Functions ---
def parse_answer(text: str | None) -> str | None:
    if not text: return None
    lower = text.lower()
    for marker in ["\n\nanswer:", "\nanswer:"]:
        idx = lower.find(marker)
        if idx != -1: return text[idx + len(marker) :].strip()
    return None

def serialize_doc(doc: dict) -> QAItem:
    return QAItem(
        id=str(doc["_id"]),
        question=doc.get("question", ""),
        text=doc.get("text"),
        answer=parse_answer(doc.get("text")),
        details=doc.get("details", {}),
        status=doc.get("status", ""),
        source=doc.get("source", ""),
        score=float(doc.get("score", 0.0) or 0.0),
    )

def _parse_golden_qa_text(text: str) -> tuple[str, str]:
    if not text: return "", ""
    if "Question:" in text and "\n\nAnswer:" in text:
        parts = text.split("\n\nAnswer:", 1)
        return parts[0].replace("Question:", "", 1).strip(), parts[1].strip()
    if "\n\n" in text:
        parts = text.split("\n\n", 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""

def get_canonical_names(user_input: str | None, mapping_type: str) -> list | None:
    if not user_input: return None
    cleaned_input = "".join(user_input.split()).lower()
    return MAPPINGS.get(mapping_type, {}).get(cleaned_input)

def _vector_search(*, collection, query_embedding: list[float], top_k: int, index_name: str, project: dict, filter_query: dict | None = None):
    vector_stage: dict = {
        "$vectorSearch": {
            "index": index_name,
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": max(top_k * 10, 50),
            "limit": top_k,
        }
    }
    if filter_query:
        vector_stage["$vectorSearch"]["filter"] = filter_query
    return list(collection.aggregate([vector_stage, {"$project": project}]))

# --- Endpoints ---
@app.post("/search", response_model=list[QAItem])
def search_questions(request: SearchRequest):
    if not request.query.strip(): raise HTTPException(status_code=400, detail="Query must not be empty.")
    
    query_text = f"Represent this sentence for searching relevant passages: {request.query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
    reviewer_filter = {}
    
    if exacts := get_canonical_names(request.crop, "crops"): reviewer_filter["details.crop"] = {"$in": exacts}
    if exacts := get_canonical_names(request.state, "states"): reviewer_filter["details.state"] = {"$in": exacts}
    if exacts := get_canonical_names(request.district, "districts"): reviewer_filter["details.district"] = {"$in": exacts}
    if exacts := get_canonical_names(request.domain, "domains"): reviewer_filter["details.domain"] = {"$in": exacts}
    if exacts := get_canonical_names(request.season, "seasons"): reviewer_filter["details.season"] = {"$in": exacts}

    results = _vector_search(
        collection=reviewer_collection, query_embedding=query_embedding, top_k=request.top_k, index_name=VECTOR_INDEX_NAME,
        project={"_id": 1, "question": 1, "text": 1, "details": 1, "status": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}},
        filter_query=reviewer_filter if reviewer_filter else None
    )
    return [serialize_doc(doc) for doc in results] if results else []

@app.post("/search_all", response_model=MultiSearchResponse)
def search_all(request: SearchRequest):
    if not request.query.strip(): raise HTTPException(status_code=400, detail="Query must not be empty.")
    
    query_text = f"Represent this sentence for searching relevant passages: {request.query}"
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    reviewer_filter = {}
    golden_filter = {}

    if exacts := get_canonical_names(request.crop, "crops"):
        reviewer_filter["details.crop"] = {"$in": exacts}
        golden_filter["metadata.Crop"] = {"$in": exacts}
    if exacts := get_canonical_names(request.state, "states"):
        reviewer_filter["details.state"] = {"$in": exacts}
        golden_filter["metadata.State"] = {"$in": exacts}
    if exacts := get_canonical_names(request.district, "districts"):
        reviewer_filter["details.district"] = {"$in": exacts}
        golden_filter["metadata.District"] = {"$in": exacts}
    if exacts := get_canonical_names(request.domain, "domains"):
        reviewer_filter["details.domain"] = {"$in": exacts}
    if exacts := get_canonical_names(request.season, "seasons"):
        reviewer_filter["details.season"] = {"$in": exacts}
        golden_filter["metadata.Season"] = {"$in": exacts}

    reviewer_raw = _vector_search(
        collection=reviewer_collection, query_embedding=query_embedding, top_k=request.top_k, index_name=VECTOR_INDEX_NAME,
        project={"_id": 1, "question": 1, "text": 1, "answer": 1, "details": 1, "status": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}},
        filter_query=reviewer_filter if reviewer_filter else None
    )
    reviewer_items = [item for item in (serialize_doc(d) for d in reviewer_raw) if item.score >= request.threshold and item.answer and item.answer.strip()]
    reviewer_items.sort(key=lambda x: x.score, reverse=True)

    golden_raw = _vector_search(
        collection=golden_qa_collection, query_embedding=query_embedding, top_k=request.top_k, index_name=GOLDEN_VECTOR_INDEX_NAME,
        project={"text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}, filter_query=golden_filter if golden_filter else None
    )
    golden_items = []
    for d in golden_raw:
        score = float(d.get("score", 0.0) or 0.0)
        if score >= request.threshold:
            q, a = _parse_golden_qa_text(d.get("text", "") or "")
            if a and a.strip():
                golden_items.append(GoldenQAItem(text=d.get("text", ""), question=q, answer=a, metadata=d.get("metadata", {}) or {}, score=score))
    golden_items.sort(key=lambda x: x.score, reverse=True)

    pop_items = []
    if not any([request.crop, request.state, request.district, request.domain, request.season]):
        pop_raw = _vector_search(
            collection=golden_pop_collection, query_embedding=query_embedding, top_k=request.top_k, index_name=GOLDEN_VECTOR_INDEX_NAME,
            project={"text": 1, "metadata": 1, "score": {"$meta": "vectorSearchScore"}}
        )
        pop_items = [PopItem(text=d.get("text", "") or "", metadata=d.get("metadata", {}) or {}, score=float(d.get("score", 0.0) or 0.0)) for d in pop_raw if float(d.get("score", 0.0) or 0.0) >= request.threshold]
        pop_items.sort(key=lambda x: x.score, reverse=True)

    return MultiSearchResponse(reviewer=reviewer_items, golden=golden_items, pop=pop_items)


@app.get("/health")
def health():
    return {"status": "ok", "mappings_loaded": {k: len(v) for k, v in MAPPINGS.items()}}


@app.get("/filters")
def get_filters():
    """Returns a clean, sorted list of available options for frontend dropdowns."""
    available_filters = {}
    
    for category, mapping in MAPPINGS.items():
        unique_options = set()
        for exact_list in mapping.values():
            if exact_list:
                # Grab the first raw string from the array to use as the display name
                unique_options.add(exact_list[0])
                
        # Sort the array alphabetically before returning it to the frontend
        available_filters[category] = sorted(list(unique_options))
        
    return available_filters


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

#uvicorn main:app --host 0.0.0.0 --port 8001