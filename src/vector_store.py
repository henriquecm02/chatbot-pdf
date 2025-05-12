import faiss
import numpy as np

def criar_indice(vetores):
    d = vetores.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(vetores)
    return index
