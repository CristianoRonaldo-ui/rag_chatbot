# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **only** from your own documents, with source citations. Built with LangChain, FAISS and OpenAI.

## How it works

1. **Load** – `DirectoryLoader` + `UnstructuredFileLoader` read every PDF in `papers/`.
2. **Split** – `RecursiveCharacterTextSplitter` cuts documents into ~1200-character chunks (200 overlap), preferring natural boundaries like headings and paragraphs.
3. **Embed & index** – each chunk is turned into a vector with OpenAI embeddings and stored in a FAISS index (cosine distance).
4. **Retrieve** – for each question, the top 5 most relevant chunks above a relevance threshold are retrieved.
5. **Generate** – `gpt-5-mini` answers using only the retrieved context, following a strict prompt:
   - answer from the provided context only
   - say *"I don't know based on the provided documents."* if the answer is not there
   - cite the source file

## Tech stack

- Python 3.12
- LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-openai`)
- FAISS (`faiss-cpu`)
- Unstructured (`unstructured[pdf,docx,pptx]`)
- OpenAI API (embeddings + `gpt-5-mini`)

## Getting started

```bash
git clone https://github.com/CristianoRonaldo-ui/rag_chatbot.git
cd rag_chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

Put your PDF files in the `papers/` folder, then run:

```bash
python chatbot.py
```

## Example

```
What is the question? How does label noise affect k-nearest neighbours?

- k = 1 (1-NN) behaves like a memorizer: its accuracy falls almost linearly
  as the label-noise rate increases (papers/Paper2_Label_Noise_Classifiers.pdf)
- Increasing k makes k-NN an "averager", which is much more robust to noise ...
```

```
What is the question? What is my name?

I don't know based on the provided documents.
```

## Sample data

The two papers in `papers/` are short synthetic studies used as demo data:

- *When Does the Bootstrap Help?* – confidence-interval coverage under small samples and skewness
- *Memorize or Average?* – how symmetric label noise affects seven standard classifiers

## Acknowledgements

Based on the tutorial ["Hướng dẫn xây dựng RAG chatbot từ A-Z"](https://www.youtube.com/watch?v=_F7Gnmk5Ta4) by Việt Nguyễn AI.