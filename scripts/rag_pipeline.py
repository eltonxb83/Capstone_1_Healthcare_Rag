#---------------------------------------
# Date          : 30 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : pathlib, pickle, faiss, numpy, sentence_transformers, llama_cpp
#Purpose        : "RAG glue code"
#Output         : Takes user query, runs retrival FAISS script, injects top k-retrieved chunks (3-5) into prompt, calls up LLM (llama.cpp GGUF 8B quantized), returns grounded answer
#Notes          :
#   1. Clinical guidlines previously processed (ie pdf text extraction, cleaning, chunking)
#   2. Pre-processing, chunking and embedding of the clinical guidelines must be completed before calling this script
#   3. FAISS retrieval (returns embeddings + ranked chunk info)
#   4. low temperature setting of 0.2 to ensure deterministic responses
#
#Flow: User Query -> retrieval_faiss.py -> top-k chunks -> prompt template -> llama-cpp.py -> grounded answer
#----------------------------------------

import pickle, faiss, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from retrieval_faiss import retrieve

#Paths
base_dir = Path(__file__).resolve().parent.parent
data_dir =base_dir/"data"/"chunks"
model_dir = base_dir/"llm"
chunk_file = data_dir/"master_chunks.pkl"
embedded_file = data_dir/"master_embeddings.npy"
faiss_file = data_dir/"faiss_index.index"
model_path = model_dir/"meta-llama-3.1-8b-instruct-q4_k_m.gguf"

top_k = 5
temperature = 0.2
max_tokens = 200
repeat_penalty = 1.25
presence_penalty = 0.3
frequency_penalty = 0.3
top_p = 0.9
mirostat_mode = 0
mirostat_tau = 0
mirostat_eta = 0

'''
temperature - sets the level of "creativity" by LLM
top_p -
    controls how the model chooses the next token. At each step, the model assigns probablities to all possible next tokens.
    There are two common filters - top-k (Keep only the opt k likely tokens) & top-p (keep the smalles set of tokens whose cumulative probability >= p)
frequency_penalty -
    Penalizes tokens based on how many times they already appeared.
presence_penalty-
    Penalizes tokens that have appeared at all.
Mirostat-
    is an adaptive sampling alogrithm. Mirostat adds feedback control to generation length and entropy.
    It tries to maintain a target surprise level(called perplexity/entropy)
'''

#Loading data/index
with open(chunk_file, "rb") as f:
    chunks = pickle.load(f)

embeddings = np.load(embedded_file)
index = faiss.read_index(str(faiss_file))

#Embedding model for query
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#LLM setup
llm = Llama(model_path = str(model_path), n_ctx=2048, n_threads =4)

#Retrieval function
def retrieval(query: str, top_k=top_k):
    results = retrieve(
        query = query,
        index = index,
        chunks= chunks,
        top_k = top_k
    )
    return results

'''
Invoking retrieve function in retrieval_faiss script
query - user_query
index - FAISS index containing embeddings for vector search
chunks- list of chunk objects aligned by position with embeddings
'''

#Prompt template
def build_prompt(context_chunks, user_question):
    context_text = "\n---\n".join([c["text"] for c in context_chunks])
    prompt = (
        "You are a healthcare assistant.\n"
        "Your task is to rewrite the provided context into an answer to the question.\n"
        "Do not add, infer, or generalise beyond the text.\n\n"
        "Rules:\n"
        "- Do NOT copy sentences verbatim unless necessary.\n"
        "- Do NOT repeat the same idea.\n"
        "- Do NOT describe the document, guideline, population, or metadata.\n"
        "- Do NOT explain what the context is.\n" 
        "- Only answer the question itself.\n"
        "- Do NOT present mulitple disconnected answers.\n"
        "- Synthesize the information into a single, fluent response.\n"
        "- If the context is fragmented, combine and clean it.\n"
        "- Only use the sentence \"The provided context does not contain a clear answer.\" if there is clearly no relevant information in the context.\n"
        "- Do not use general medical knowledge outside the context\n"
        "- Every statement must be directly supported by the provided context.\n"
        "- Do not infer, assume, or add clinical facts not explicitly present.\n"
        "- Do NOT add notes, explanations about your behavior, or meta commentary.\n "
        "- Cite the source(s) after your answer.\n"
        "- If a sentence's source cannot be cited, do not include it.\n"
        "- Keep the answer concise, clinically clear, and well structured.\n"
        "- Write the answer in a single concise paragraph.\n"
        "Answer style:\n"
        "- Professional clinical tone.\n"
        "- Explain rather than quote.\n"
        "- No repetition.\n"
        "- Limit your answer to 120 words.\n"
        "- Trim ALL incomplete sentences.\n"
        "Here is the retrieved context:\n"
        "<CONTEXT>\n"
        f"{context_text}"
        "</CONTEXT>\n"
        f"Question: {user_question}\n"
        "Answer:\n"
        "<ANSWER>\n"
    )
    return prompt

#RAG pipeline
def run_rag(query: str):
    retrieved = retrieval(query, top_k=top_k)
    if not retrieved:
        return "No relevant context found. I don't know."
    
    #temporary debug block
    print("\n--- Retrieved Chunks ---\n")
    for r in retrieved:
        print(f"[Rank {r['rank']}] Distance : {r['distance']:.4f}")
        print(f"Source: {r['metadata']['source']}")
        print(f"Population: {r['metadata']['population']}")
        print(f"Section: {r['metadata']['section']}")
        print(f"Text:\n{r['text'][:500]}\n")
        print("-"*80)

    prompt = build_prompt(retrieved, query)
    response = llm(prompt=prompt, max_tokens=max_tokens, temperature= temperature, repeat_penalty = repeat_penalty, top_p = top_p, presence_penalty= presence_penalty, frequency_penalty= frequency_penalty, mirostat_mode = mirostat_mode, mirostat_tau = mirostat_tau, mirostat_eta= mirostat_eta, stop=["\n\n---", "\n\nQuestion:", "</ANSWER>"])
    answer = response["choices"][0]["text"]
    return {
        "answer": answer,
        "chunks" : retrieved
    }

#test/demo
# if __name__ == "__main__":
#     user_query = "How is sepsis recognised in children?"
#     answer = run_rag(user_query)
#     print("\n---RAG Answer---\n")
#     print(answer)