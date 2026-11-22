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

`ingest.py` only needs to run once, or again if the PDFs change. It also copies the PDFs to `static/` so the source links can open them.

## Features

- Due date from the first day of the last period (Naegele's rule), used for trimester-aware answers
- Clickable sources that open the PDF at the right page
- Grounding / hallucination score, confidence score and answer metrics under each answer
- English, Hindi and Bengali (danger-word check before and after translation)
- Voice questions (Groq Whisper) and a Listen button (gTTS); both need internet

## Evaluation

```
python evaluate.py final
```

Checks the red-flag filter, retrieval and refusals on the 45 questions in `eval_questions.csv` (no LLM calls) and saves the numbers to `results.csv`.

```
python evaluate.py --llm
```

Runs 10 questions through the real LLM (about 5 minutes) and saves grounding, keyword and latency results to `answer_results.csv`.

## Disclaimer

This is a student project. It is not medical advice and does not replace a doctor.
