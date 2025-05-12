from sentence_transformers import SentenceTransformer

def gerar_embeddings(sentencas):
    modelo = SentenceTransformer('all-MiniLM-L6-v2')
    vetores = modelo.encode(sentencas)
    return vetores
