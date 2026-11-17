# 🤖 Chatbot PDF (protótipo de estudo)

> **📦 Repositório arquivado.** Este foi meu primeiro experimento com embeddings e busca vetorial.
> A versão que substitui este projeto, com recuperação medida, citações verificadas e testes, é o
> **[ai-knowledge-assistant](https://github.com/henriquecm02/ai-knowledge-assistant)**.

## O que é

Protótipo que extrai o texto de um PDF, gera embeddings das sentenças (Sentence Transformers), cria um índice FAISS e responde perguntas com um modelo de QA extrativo (DistilBERT) em uma interface Streamlit.

## Limitações conhecidas

Mantive o código como estava para registrar o ponto de partida. Os problemas abaixo foram o que me levou a refazer o projeto:

- **O índice vetorial não participa da resposta.** O FAISS é criado, mas a pergunta nunca é buscada nele: o modelo de QA recebe o texto inteiro do PDF como contexto.
- **Falta `import numpy as np` em `src/chatbot_app.py`**, então o app falha ao carregar um PDF.
- **Segmentação ingênua:** o texto é dividido por `"."`, o que quebra siglas, valores e decimais.
- **Sem fonte na resposta** (documento, seção ou página) e sem tratamento para pergunta sem resposta no texto.
- **Sem testes e sem avaliação** de qualidade da busca.

## O que o ai-knowledge-assistant corrige

| Aqui | No [ai-knowledge-assistant](https://github.com/henriquecm02/ai-knowledge-assistant) |
|---|---|
| Sentenças cortadas em `"."` | Chunking por seção e página |
| Só embeddings, índice sem uso | Busca híbrida BM25 + embeddings com RRF |
| Resposta sem fonte | Citações com documento, seção e página, verificadas em código |
| Responde sempre | Diz "não encontrei" quando a informação não está nos documentos |
| Sem medição | Avaliação da recuperação antes de qualquer geração |

## Tecnologias

Python · Streamlit · PyMuPDF · Sentence Transformers · FAISS · Transformers

## Como rodar (referência)

```bash
pip install -r requirements.txt
streamlit run src/chatbot_app.py
```

Para rodar de fato é preciso adicionar `import numpy as np` no topo de `src/chatbot_app.py`.
