# Retrieval-Augmented AI Teaching Assistant

A modular Python assistant that answers student questions using your own lecture notes by combining keyword (BM25) and semantic (embedding-based) search with retrieval-augmented generation (RAG) via advanced language models.

---

## Features

- Hybrid search: BM25 keyword + semantic vector retrieval
- Retrieval-augmented generation with OpenAI GPT models
- Customizable prompts for different student levels
- Supports text and optional image/diagram retrieval
- Modular and extensible design

---

## Project Structure

```
RAG_TA_project/
├── app/
│   ├── main.py
│   ├── ingest.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── multimodal.py
│   └── config/
│       └── prompts.yaml
├── data/
│   └── raw/
├── .env
├── requirements.txt
└── README.md
```


---

## Quickstart

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/RAG_TA_project.git
cd RAG_TA_project
```

2. **Set up the environment**

```bash
python -m venv .venv
# Activate:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

3. **Add your lecture materials** to `data/raw/` (txt, pdf, pptx, images).
4. **Set your OpenAI API key** in a `.env` file:

```
OPENAI_API_KEY=sk-...your-key...
```

5. **Run the assistant**

```bash
python -m app.main
```


---

## Requirements

- Python 3.9–3.11 recommended
- OpenAI API key
- See `requirements.txt` for dependencies

---

## Customization

- Edit `app/config/prompts.yaml` for prompt styles.
- Change embedding or LLM models in `retrieval.py` and `generation.py`.
- Add file readers in `ingest.py` for new formats.

---
