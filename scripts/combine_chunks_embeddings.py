#---------------------------------------
# Date          : 30 Dec 25
#Author         : Elton Tay, Chatgpt as part of AI driven solution
#Dependencies   : pathlib, pickle, numpy, faiss
#Purpose        : Combining individual .pkl and .npy files into a single master .npy, .pkl and index file.
#Output         : Master .pkl file, .npy file and index file
#----------------------------------------

from pathlib import Path
import pickle, numpy as np, faiss

chunk_dir = Path("data/chunks")
master_chunk_file = chunk_dir/"master_chunks.pkl"
master_embed_file = chunk_dir/"master_embeddings.npy"
faiss_index_file = chunk_dir/"faiss_index.index"

all_chunks =[]
all_embeddings =[]

#Loop over all chunk/embedging files
for chunk_file in chunk_dir.glob("*_chunks.pkl"):
    stem_name = chunk_file.stem.replace("_chunks","")
    embeded_file = chunk_dir/f"{stem_name}_embeddings.npy"

    #Load chunks
    with open(chunk_file, "rb") as f:
        chunks = pickle.load(f)

    #Load embeddings
    embeddings = np.load(embeded_file)

    #Append to master lists
    all_chunks.extend(chunks)
    all_embeddings.append(embeddings)

'''
append()
. Adds a single object to the end of a list, whatever the object is
. The object is added as-is, so if you append a listm you get a nested list

extend()
. Takes an iterable (list, tuple, etc) and adds all its elements individually to the list
. It "flattens" one level of the iterable into the list
'''

all_embeddings = np.vstack(all_embeddings)

#Save master files
with open(master_chunk_file, "wb") as f:
    pickle.dump(all_chunks, f)

np.save(master_embed_file, all_embeddings)

#Build FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(all_embeddings)

#Save FAISS index
faiss.write_index(index, str(faiss_index_file))
'''
faiss.write_index ->FAISS function to save and index to disk
str(faiss_index_file) -> Path to the file you want to save to, It must be a string.
'''

print(f"Master chunks: {master_chunk_file}")
print(f"Master embeddings: {master_embed_file}")
print(f"FAISS index: {faiss_index_file}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Master index creation complete!")