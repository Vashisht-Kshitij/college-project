# PregnaCare AI

An evidence-grounded pregnancy guidance chatbot using RAG. It answers only from the antenatal guideline PDFs in `data/`.

## Setup

```
pip install -r requirements.txt
```

Create a `.env` file in this folder with your Groq key:

```
GROQ_API_KEY=your_key_here
```

## Run

```
python ingest.py
streamlit run app.py
```

`ingest.py` only needs to run once, or again if the PDFs change.

## Disclaimer

This is a student project. It is not medical advice and does not replace a doctor.
