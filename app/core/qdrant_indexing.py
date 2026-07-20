"""
Pipeline stage: Vector indexing and semantic search using Qdrant.
(Replaces an earlier Chroma-based prototype, since removed.)

Two-stage retrieval: query_points() does fast approximate nearest-neighbor
search over embeddings to get a candidate pool, then a cross-encoder
re-ranks just those candidates for higher precision. This is standard
practice for RAG systems -- embeddings are fast but approximate, rerankers
are accurate but too slow to run over the full collection.
"""
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

# All videos and all users share ONE Qdrant collection. Isolation between
# users/videos is enforced entirely via payload filters (video_id, owner_id)
# at query time -- not via separate collections. Anyone querying Qdrant
# directly without a filter would see all indexed content across all users.
COLLECTION_NAME = "video_search"
EMBEDDING_DIM = 384
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE embedding models are trained asymmetrically: search QUERIES need this
# instruction prefix prepended, but indexed DOCUMENTS (transcript/caption
# chunks) do not. This is intentional and required for good retrieval
# quality -- see its use in search() below vs. its absence in index_video().
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20  # how many candidates the embedding search retrieves
                         # before reranking narrows down to n_results

_client = None
_embedder = None
_reranker = None


def _get_client():
    """Lazily create (and cache) the Qdrant client, ensuring the collection
    exists on first use."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client):
    """Create the shared collection if it doesn't exist yet (idempotent --
    safe to call on every startup)."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'")


def _get_embedder():
    """Lazily load (and cache) the sentence embedding model."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL})...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_reranker():
    """Lazily load (and cache) the cross-encoder reranking model."""
    global _reranker
    if _reranker is None:
        print(f"Loading re-ranking model ({RERANK_MODEL})...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def build_chunks(transcript, captions, video_id, owner_id):
    """
    Convert transcript segments and visual captions into a flat list of
    indexable "chunks" -- one per speech segment and one per captioned
    keyframe, each tagged with type ("speech"/"visual"), timestamp,
    video_id, and owner_id for later filtering.
    """
    chunks = []
    for seg in transcript:
        # Prefix with speaker label when available (post-diarization),
        # so search results show who said what.
        speaker_prefix = f"{seg['speaker']}: " if "speaker" in seg else ""
        chunks.append({
            "text": f"{speaker_prefix}{seg['text']}",
            "timestamp": seg["start"],
            "type": "speech",
            "video_id": video_id,
            "owner_id": owner_id,
        })
    for cap in captions:
        chunks.append({
            "text": cap["caption"],
            "timestamp": cap["timestamp"],
            "type": "visual",
            "video_id": video_id,
            "owner_id": owner_id,
        })
    return chunks


def index_video(transcript, captions, video_id, owner_id):
    """
    Embed and upsert all chunks for a video into Qdrant.

    NOTE: no de-duplication -- each call generates fresh uuid4 point IDs,
    so calling this twice for the same video_id (e.g. on reprocessing)
    creates duplicate entries rather than overwriting the previous ones.
    If reprocessing becomes a supported flow, delete-by-video_id before
    re-indexing to avoid duplicate search results.
    """
    chunks = build_chunks(transcript, captions, video_id, owner_id)
    if not chunks:
        print("No chunks to index.")
        return

    client = _get_client()
    embedder = _get_embedder()

    # NOTE: no QUERY_PREFIX here -- documents are embedded "as-is", which
    # is correct for this asymmetric embedding model (see QUERY_PREFIX note
    # above). Only queries in search() get the prefix.
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts).tolist()

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload=chunks[i],
        )
        for i in range(len(chunks))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(chunks)} chunks for video '{video_id}' into Qdrant.")


def search(query, n_results=5, video_id=None, owner_id=None):
    """
    Semantic search over indexed chunks.

    Optionally scoped to a specific video_id and/or owner_id via payload
    filters -- this is the ONLY mechanism enforcing that a user's search
    only sees their own (or a specific video's) content, since all data
    lives in one shared Qdrant collection.

    Two-stage retrieval:
      1. Embed the query (with QUERY_PREFIX) and fetch RERANK_CANDIDATES
         nearest neighbors from Qdrant -- fast but approximate.
      2. Re-score just those candidates with a cross-encoder, which looks
         at the query and each candidate text together (more accurate,
         too slow to run over the whole collection) -- then return the
         top n_results by that more accurate score.
    """
    # SECURITY: owner_id is required and enforced here, not left optional.
    # This is the ONLY thing preventing one user's search from returning
    # another user's indexed content (all videos share one Qdrant
    # collection -- see COLLECTION_NAME note above). Failing loudly on a
    # missing owner_id is safer than silently searching across all users,
    # which is what happened before this check existed whenever a caller
    # passed owner_id=None or "".
    if not owner_id:
        raise ValueError("search() requires a non-empty owner_id -- refusing to "
                          "search without a user scope, since this collection is "
                          "shared across all users.")

    client = _get_client()
    embedder = _get_embedder()

    query_vector = embedder.encode([QUERY_PREFIX + query])[0].tolist()

    conditions = [FieldCondition(key="owner_id", match=MatchValue(value=owner_id))]
    if video_id:
        conditions.append(FieldCondition(key="video_id", match=MatchValue(value=video_id)))
    query_filter = Filter(must=conditions)

    # Fetch more candidates than we'll ultimately return (RERANK_CANDIDATES),
    # so the reranker has a meaningful pool to re-sort rather than just the
    # final n_results from the (less accurate) embedding search alone.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=max(n_results, RERANK_CANDIDATES),
    )

    candidates = [
        {
            "text": r.payload["text"],
            "timestamp": r.payload["timestamp"],
            "type": r.payload["type"],
            "video_id": r.payload["video_id"],
            "owner_id": r.payload.get("owner_id"),
            "score": r.score,  # original embedding similarity score
        }
        for r in response.points
    ]

    if not candidates:
        return []

    # Re-rank: score each (query, candidate_text) pair with the cross-encoder
    # and re-sort by that score instead of the original embedding score.
    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    for c, score in zip(candidates, rerank_scores):
        c["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:n_results]
