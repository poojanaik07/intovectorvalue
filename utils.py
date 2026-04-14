import numpy as np
import faiss

def find_duplicates_faiss(embeddings, threshold=0.7):
    # Convert to numpy float32 (FAISS requirement)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize vectors (important for cosine similarity)
    faiss.normalize_L2(embeddings)

    # Create index
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product

    # Add data
    index.add(embeddings)

    # Search top K similar
    k = 5  # number of neighbors
    distances, indices = index.search(embeddings, k)

    # Grouping
    groups = [-1] * len(embeddings)
    group_id = 0

    for i in range(len(embeddings)):
        if groups[i] != -1:
            continue

        groups[i] = group_id

        for j, sim in zip(indices[i], distances[i]):
            if sim >= threshold:
                groups[j] = group_id

        group_id += 1

    return groups
