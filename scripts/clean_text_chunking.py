#---------------------------------------
# Date          : 23 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : pathlib, regex, sentence_transformers, nltk, numpy, pickle
#Purpose        : Peform overlapping chunking, addition of metadat and embedding on cleaned text extracted from NICE clinical guidelines
#Output         : embeddings.npy (contains L2 vectors) and corresponding .pkl files (containing chunks and respective metadata)
#----------------------------------------
from pathlib import Path
import re, nltk,  numpy as np, pickle
from sentence_transformers import SentenceTransformer

'''
NLTK
nltk = Natural Language Toolkit
.Python library for working with human language text
.Provides tokenization, taggin, parsing, and other NLP utilities

Purpose - to break a long guideline into sentences, so that chunks do not break in the middle of a sentence, which preserve semantic meaning
Without sentence tokenization, chunking may cut a sentence in half, which can confuse the LLM during RAG retrieval.

punkt - pre-trained sentence tokenizer model in NLTK
it is a pre-trained rule-based model that knows how sentences end in real langauge.

Pickle
pickle = Python Library for Serializing Python Objects
Serization = converting a Python object (list, dict etc) into a byte stream that can be saved to disk
We can load it back into Python exactly as it was

Pipeline stages
Text splitting Stage - Tool : NLTK - Splits text into sentences (via sentence tokenization)
Chunking - Tool : Python code - Group and overlap sentences
Chunk Persistence - Tool: pickle - Saves Python objects
Semantic encoding - Tool: all-MiniLM-L6-V2 - Converts text -> vectors
'''
#nltk.download('punkt_tab')
# Run once to download sentence tokenizer

clean_text_dir = Path("data/cleaned")
chunk_dir = Path("data/chunks")
chunk_dir.mkdir(parents=True, exist_ok=True)

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

max_chars = 600
overlap_sentences = 2

def infer_population(text:str):
    text_lower = text.lower()
    if "under 16" in text_lower or "children" in text_lower:
        return "under 16"
    if "16 or over" in text_lower or "aged 16" in text_lower:
        return "over 16"
    return "unknown"

def infer_section(text: str):
    text_lower = text.lower()
    if "overview" in text_lower:
        return "overview"
    if "recognition" in text_lower:
        return "recognition"
    if "assessment" in text_lower:
        return "assessment"
    if "treatment" in text_lower:
        return "treatment"
    return "unknown"

def chunk_text(text: str, source_name: str, max_char: int = max_chars, overlap: int = overlap_sentences):
    sentences = nltk.tokenize.sent_tokenize(text) #Scans the text, detects sentence boundaries, outputs a python list of sentence strings
    chunks = []
    current_chunk = []

    for sentence in sentences:
        candidate = " ".join(current_chunk + [sentence])

        if len(candidate) <= max_char:
            current_chunk.append(sentence)
        else:
            chunk_text = " ".join(current_chunk)

            chunk = {
                "text": chunk_text,
                "source": source_name,
                "population": infer_population(chunk_text),
                "section": infer_section(chunk_text)
            }
            chunks.append(chunk)
        
            #overlap handling
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk.append(sentence)

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunk = {
            "text": chunk_text,
            "source": source_name,
            "population": infer_population(chunk_text),
            "section": infer_section(chunk_text)
        }
        chunks.append(chunk)

    return chunks

'''
candidate = " ".join(current_chunk + [sentence])
.concatenation of current_chunk and sentence. Sentence within sqaure paratheses as nltk.tokenize.sent_tokenize(text) returns a list of strings

if len(candidate) <=max_char:
    current_chunk.append(sentence)

. if candidate is less or equal max_char, current sentence is appended to current_chunk
. loop next sentence till candidate is more than max_char then corresponding else statement will trigger

else:
    chunk_text = " ".join(current_chunk)

    chunk = {
        "text" : chunk_text,
        "source": source_name,
        "population": infer_population(chunk_text),
        "section": infer_section(chunk_text)
    }
    chunks.append(chunk)

    current_chunk = current_chunk[-overlap:] if overlap > 0 else []
    current_chunk.append(sentence)

. when len(cadidate) > max_char, current_chunk assign to chunk_text
. preliminary metadata values assigned based on predefined key:value pairs
. append chunk to chunks dictionary list
. current_chunk gets re-assigned with last two setences for overlap
. current sentence gets appended to current_chunk
. gets loop back to the begining with next lopped sentence. Loop continues till last sentence of the function

if current_chunk:
    chunk_text = " ".join(current_chunk)
    chunk = {
        "text": chunk_text,
        "source": source_name,
        "population": infer_population(chunk_text),
        "section": infer_section(chunk_text)
    }
    chunks.append(chunk)
. The last sentence gets added to the candidate and if statements still holds true and the last sentence gets appended to current_chunk
. Returns back to the for loop for the next sentence and returns empty as no further sentence. this stops the if-else sequence
. this triggers the last "if" code block to capture the remaining sentences and append to the chunks.

'''

def main():
    for txt_file in clean_text_dir.glob("*.txt"):
        print(f"Processing: {txt_file.name}")

        source_name = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, source_name=source_name)

        print(f"->{len(chunks)} chunks created")

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar = True)
        embeddings = np.array(embeddings)

        chunk_file = chunk_dir/f"{txt_file.stem}_chunks.pkl"
        with open(chunk_file, "wb") as f: # "wb" <- binary write mode
            pickle.dump(chunks, f)

        embed_file = chunk_dir/f"{txt_file.stem}_embeddings.npy"
        np.save(embed_file, embeddings)

        print(f"Saved chunks to: {chunk_file}")
        print(f"Saved embeddings to: {embed_file}\n")

if __name__ == "__main__":
    main()