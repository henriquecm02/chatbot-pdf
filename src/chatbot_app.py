import streamlit as st
from extract_text import extrair_texto_pdf
from generate_embeddings import gerar_embeddings
from vector_store import criar_indice
from transformers import pipeline

st.title("📚🤖 Chatbot de PDF com IA")

arquivo_pdf = st.file_uploader("Envie seu PDF", type="pdf")
if arquivo_pdf:
    texto = extrair_texto_pdf(arquivo_pdf)
    sentencas = texto.split(".")
    vetores = gerar_embeddings(sentencas)
    indice = criar_indice(np.array(vetores))

    pergunta = st.text_input("Digite sua pergunta:")
    if pergunta:
        modelo_qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        resultado = modelo_qa(question=pergunta, context=texto)
        st.write("🤖 Resposta:", resultado['answer'])
