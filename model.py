from sentence_transformers import SentenceTransformer

# Load once (important)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_embeddings(text_list):
    return model.encode(text_list, batch_size=128, show_progress_bar=True)
