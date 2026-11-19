# Streamlit chat UI and the main question -> answer flow
import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from safety import ESCALATION_MESSAGE, SAFE_MESSAGE, get_trimester, get_week, is_red_flag, is_safe_answer

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.40
SOURCE_NAMES = {
    "who_pregenant_woman.pdf": "WHO ANC Recommendations (2016)",
    "Care During Pregnancy and Childbirth Training Manual for CHO at AB-HWC.pdf": "MoHFW CHO Training Manual (2021)",
    "zaf-mn-21-01-guideline-2017-eng-principles-of-good-care---guidelines.pdf": "South Africa Maternity Care Guidelines (2017)",
}
NOT_ENOUGH_INFO ="I don't have enough verified information on this in my knowledge base. Please consult your doctor."

SYSTEM_PROMPT = """You are PregnaCare AI, a pregnancy information assistant.

Rules:
- Answer ONLY from the given evidence. Do not use outside knowledge.
- If the evidence does not clearly answer the question, say you are not sure.
- Never diagnose the user.
- Never prescribe or mention medicine doses or amounts.
- Keep the answer short and in simple language.
- The evidence may be written for health workers. Rewrite it for a pregnant woman.

Use exactly this format and do not list sources (the app adds them):
**Answer:** <your answer>

**When to see a doctor:** <what the evidence says. If it says nothing, write: Talk to your doctor if you are worried or notice anything new.>"""

# the model ignored the rules when they were only in the system prompt, so repeat them after the question
REMINDER = """Reminder: every sentence of your answer must come from the evidence above.
Do not name any activity, food or symptom that is not written in the evidence.
Do not give examples of your own. Only use a "such as" list if those exact items are in the evidence.
If the evidence does not list specific activities or foods, say the guidelines do not list them.
If the sources disagree, follow the WHO ANC Recommendations and say that this is what WHO recommends."""


@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = chromadb.PersistentClient(path="chroma_db").get_collection("pregnancy_docs")
    return model, collection


def retrieve(question, trimester):
    model, collection = load_resources()
    query = f"{question} ({trimester} trimester of pregnancy)" if trimester else question
    embedding = model.encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=TOP_K)

    chunks = []
    for text, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        source = SOURCE_NAMES.get(meta["source"], meta["source"])
        chunks.append({"text": text, "source": source, "page": meta["page"], "similarity": 1 - distance})
    return chunks


def best_score(chunks):
    # the threshold is checked against this: cosine similarity between the question and the best chunk
    return max(c["similarity"] for c in chunks)


def ask_llm(question, trimester, chunks):
    evidence = "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks)
    stage = f"The user is in the {trimester} trimester.\n\n" if trimester else ""
    response = Groq().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        reasoning_effort="low",  # otherwise it can use all its tokens thinking and return nothing
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{stage}Evidence:\n{evidence}\n\nQuestion: {question}\n\n{REMINDER}"},
        ],
    )
    return response.choices[0].message.content


def format_sources(chunks):
    sources = dict.fromkeys(f"- {c['source']}, page {c['page']}" for c in chunks)  # removes duplicates
    return "**Sources:**\n" + "\n".join(sources)


def find_evidence(question, sidebar_week):
    # decides which path runs (red_flag / refused / answer); evaluate.py uses this too
    if is_red_flag(question):
        return "red_flag", None, []  # safety check comes first, nothing else runs

    week = get_week(question) or sidebar_week
    trimester = get_trimester(week) if week else None
    chunks = retrieve(question, trimester)
    path = "refused" if best_score(chunks) < SIMILARITY_THRESHOLD else "answer"
    return path, trimester, chunks


def get_response(question, sidebar_week):
    # returns the answer, the evidence chunks and a caption saying which path ran
    path, trimester, chunks = find_evidence(question, sidebar_week)
    if path == "red_flag":
        return ESCALATION_MESSAGE, [], "Red-flag escalation"
    if path == "refused":
        return NOT_ENOUGH_INFO, chunks, f"Refused: not enough evidence (best score {best_score(chunks):.2f})"

    try:
        answer = ask_llm(question, trimester, chunks)
    except Exception as e:
        return f"Could not reach the AI service, please try again. ({e})", chunks, "LLM call failed"

    if not answer:
        return "I could not generate an answer, please try again.", chunks, "LLM returned no answer"
    if not is_safe_answer(answer):
        return SAFE_MESSAGE, chunks, "Answer blocked by safety validation"
    return f"{answer}\n\n{format_sources(chunks)}", chunks, f"Answered from {len(chunks)} evidence chunks"


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])
        if message.get("chunks"):
            with st.expander("Retrieved evidence"):
                for c in message["chunks"]:
                    st.markdown(f"**{c['source']}, page {c['page']}** (similarity {c['similarity']:.2f})")
                    st.text(c["text"])


# streamlit runs this file as __main__; the check lets evaluate.py import the functions above
if __name__ == "__main__":
    st.title("🤰 PregnaCare AI")
    st.warning(
        "This chatbot gives general information from WHO and government antenatal guidelines. "
        "It is not a doctor and does not replace medical advice. In an emergency, call 108."
    )

    sidebar_week = st.sidebar.number_input("Pregnancy week (optional, 0 = not set)", min_value=0, max_value=42, value=0)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        show_message(message)

    if question := st.chat_input("Ask a question about your pregnancy"):
        st.session_state.messages.append({"role": "user", "content": question})
        show_message(st.session_state.messages[-1])

        with st.spinner("Checking the guidelines..."):
            answer, chunks, caption = get_response(question, sidebar_week)

        message = {"role": "assistant", "content": answer, "chunks": chunks, "caption": caption}
        st.session_state.messages.append(message)
        show_message(st.session_state.messages[-1])
