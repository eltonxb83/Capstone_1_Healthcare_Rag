#---------------------------------------
# Date          : 24 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : faiss, pathlib, pickle, numpy, sentence_transformers
#Purpose        : 
#   1. FAISS retrieval based on vector similarity
#   2. Post-FAISS retrieval refinement using chunk-level metadata
#Output         : Top-k retrieved chunks (tex+metadata) for downstram RAG prompting
#Notes          : 
#   1. Invoked by rag_pipeline.py prior to LLM prompt construction
#   2. Supports population and section-awre retrieval
#----------------------------------------

from pathlib import Path
import pickle, numpy as np, faiss
from sentence_transformers import SentenceTransformer

chunk_dir = Path("data/chunks")

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

def infer_query_intent(query: str):
    q = query.lower()

    population = None
    if "child" in q or "children" in q or "under 16" in q:
        population = "under 16"
    elif "adult" in q or "over 16" in q:
        population = "over 16"

    section = None
    if "symptom" in q:
        section = "recognition"
    elif "diagnosis" in q:
        section = "assessment"
    elif "treat" in q or "manage" in q:
        section = "treatment"
    
    return population, section

def load_chunks_and_embeddings(stem_name: str):
    chunk_file = chunk_dir/f"{stem_name}_chunks.pkl"
    embed_file = chunk_dir/f"{stem_name}_embeddings.npy"

    with open(chunk_file, "rb") as f:
        chunks = pickle.load(f)

    embeddings = np.load(embed_file)

    return chunks, embeddings

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

'''
After chunking script (ie embeddings = model.encode(chunks)), the shape is
(number_of_chunks, embedding_dimension). For all-MiniLM-L6-v2, embedding_dimension = 384, number_of_chunks = N (depends on the document length)
Each row = One Chunk
Each column = one semantic feature

index = faiss.IndexFlatL2(dimension)
It creates an empty index structure, configures it to:
    . Expect vectors of length dimensions
    . use L2 (Euclidean) distance for similarity
Note:
    1. We are effevtively using cosine similarity, even though the FAISS index says L2 because of how sentence-transformer embeddings behave
    2. SentenceTransformer embeddings are already L2-normalized (Perform normalization action here for future-proofing - ie change in embedding model). L2 distance == cosine similarity (when normalized)

index.add(embeddings)
This is where data actually enters FAISS
It only stores the data, no math calculations happening here.

Note- All similarity/distance math happens at index.search()

Note (Cosine Similarity)
For two vectors a and b:

                a⋅b=∥a∥∥b∥cos(θ)

After normalization (where absolute value of a and b becomes 1):

                    a⋅b=cos(θ)

Angle θ     cos(θ)           Meaning
0°           1.0             Same direction → most similar
30°          0.87	         Very similar
90°          0.0	         Orthogonal → unrelated
120°	     -0.5	         Opposite-ish
180°         -1.0            Opposite direction

Larger dot product = smaller angle = more similar

Note (Euclidean Distance Calculations-Normalized)
For two vectors a and b (normalized):
                    L2^2 = 2-2cos(θ)

Cosine similarity      Normalized L2²      Interpretation
1.0                         0	            Identical
0.8                         0.4	            Very similar
0.5	                        1.0	            Moderately  similar
0.0                         2.0	            Orthogonal
-1.0	                    4.0	            Opposite

FAISS - finds the smallest L2 distance which is equivalent to finding the largest cosine similarity
'''

def retrieve(query: str, index, chunks, top_k: int =5, faiss_k: int =15):
    query_embedding = model.encode([query], normalize_embeddings =True)
    distances, indices = index.search(query_embedding, faiss_k)

    desired_population, desired_section = infer_query_intent(query)

    scored_results = []

    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]

        if not isinstance(chunk, dict):
            print("Non-dict chunk detected:", type(chunk), "example:", chunk[:120])
            raise TypeError("Chunk is not a dictionary")

        score = -distances[0][rank]

        if desired_population and chunk.get("population") == desired_population:
            score += 0.3
        
        if desired_section and chunk.get("section") == desired_section:
            score += 0.3
        
        scored_results.append({
            "score": score,
            "distance": distances[0][rank],
            "chunk": chunk
        })
    
    scored_results.sort(key = lambda x: x["score"], reverse = True)

    results = []
    for i, r in enumerate(scored_results[:top_k]):
        results.append({
            "rank": i+1,
            "distance": r["distance"],
            "text": r["chunk"]["text"],
            "metadata":{
                "source": r["chunk"]["source"],
                "population": r["chunk"]['population'],
                "section": r["chunk"]["section"]
            }
        })

    return results

'''
def retrieve(query: str, index, chunks, top_k: int = 5, faiss_k: int = 15)
user-defined retrieve function with query being the embedded user query,
index being the faiss index built earlier on the stored data (.npy file), chunks being the actual strings stored (.pkl file),
top_k being the top hits in terms of Euclidean distance (smallest).
faiss_k returns top 15 chunks and top_k is selected from there

query_embeddings = model.encode([query])
embedding of user query

distance, indices = index.search(query_embedding, faiss_k)
FAISS search conduct here
assigning Euclidean distance values to distance
assigning indices of corresponding top-hits

Note-
FAISS is not designed for "one query -> one result"
FAISS is designed for "many queries at once -> many nearest neighbors per query"
FAISS is always thinking in terms of (number_of_queries, top_k results)
FAISS always returns two arrays- 1. distances 2. indices. Both are 2-d Numpy arrays with shape
(number_of_queries,top_k). ie - distance.shape == (1,5) and indices.shape ==(1,5)

enumerate(indices[0])
enumerate function is a built-in function that adds a counter to an iterable (like a list, tuple or string)
and returns it as an enumerate object.
What it actually does - it takes one row of shape (5,) and turns it into 5 python pairs (rank, idx)
indices[0] <- changes from (1,5) to (5,). 1-D iterable
enumerate() does not change dimensionality. It wraps each element with a counter. We get 5 iterations for this case.
Each iteraion yields one tuple of length 2


rank, idx in enumerate(indices[0])
rank registers the counter created by the enumerate function
idx registers the actual corresponding indices which to be used to retrieve the stored chunks

Note-
FAISS is designed for batching. That is why indices[0] exist although in practice is redundant for this case
as there is only a single query

score = -distances[0][rank]
FAISS returns distance, not similarity
    . Smaller distance  -> more similar
    . Larger distance -> less similar
But scoring systems work the opposite way:
    . Higher score -> better
    . Lower score -> worse
Note:
    . Converting distance -> score
    . To allow adding of metadata scoring later

'''
def main():
    """
    Test harness for validating FAISS retrieval
    Not used in production pipelines.
    """
    stem_name = "NICE_Sepsis_Over_16"

    chunks, embeddings = load_chunks_and_embeddings(stem_name)
    index = build_faiss_index(embeddings)

    print(f"Loaded {len(chunks)} chunks into FAISS index\n")

    query = "What are the symptoms of sepsis in children?"
    results = retrieve(query, index, chunks, top_k=5)

    print(f"Query: {query}")
    print("-"*80)

    for r in results:
        print(f"[Rank {r['rank']}] Distance : {r['distance']: .4f}")
        print(r["text"])
        print("-"*80)

if __name__ == "__main__":
    main()