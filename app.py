# Streamlit chat UI and the main question -> answer flow
import io
import os
import re
import time
from datetime import date, timedelta
from urllib.parse import quote

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from safety import (ESCALATION_MESSAGE, LOCAL_MESSAGES, SAFE_MESSAGE, get_trimester, get_week, is_red_flag,
                    is_safe_answer)

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
TOP_K = 4
SIMILARITY_THRESHOLD = 0.40
# a sentence of the answer is "supported" if it is at least this similar to one of the 4 chunks
# (picked on real answers vs made-up sentences; see the report)
GROUNDING_CUTOFF = 0.58
SOURCE_NAMES = {
    "who_pregenant_woman.pdf": "WHO ANC Recommendations (2016)",
    "Care During Pregnancy and Childbirth Training Manual for CHO at AB-HWC.pdf": "MoHFW CHO Training Manual (2021)",
    "zaf-mn-21-01-guideline-2017-eng-principles-of-good-care---guidelines.pdf": "South Africa Maternity Care Guidelines (2017)",
}
NOT_ENOUGH_INFO = "I don't have enough verified information on this in my knowledge base. Please consult your doctor."
LANGUAGES = {"English": "en", "हिन्दी (Hindi)": "hi", "বাংলা (Bengali)": "bn"}
LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}
AVATARS = {"user": "🤰", "assistant": "🩺"}
BADGE_COLORS = {"High": "green", "Medium": "orange", "Low": "red"}

# soft rounded chat bubbles + a small header animation (switched off for people who prefer reduced motion)
PAGE_CSS = """
<style>
[data-testid="stChatMessage"] {background: #ffffff; border-radius: 18px; padding: 0.8rem 1rem;
    box-shadow: 0 2px 10px rgba(201, 85, 122, 0.12); margin-bottom: 0.6rem;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #fde8e4, #fff3ea);}
[data-testid="stChatInput"] textarea {border-radius: 14px;}
.pc-header {text-align: center; margin-top: -1rem;}
.pc-title {font-size: 2.2rem; font-weight: 700; color: #b0466c;}
.pc-sub {color: #7a6066; font-size: 0.95rem;}
.pc-heart {display: inline-block; animation: pc-beat 1.6s ease-in-out infinite;}
.pc-track {position: relative; height: 34px; max-width: 420px; margin: 0.2rem auto 0.8rem;
    border-bottom: 2px dashed #f2b8c6;}
.pc-baby {position: absolute; bottom: 2px; left: 0; font-size: 1.5rem;
    animation: pc-crawl 14s ease-in-out infinite alternate;}
@keyframes pc-beat {0%, 100% {transform: scale(1);} 15% {transform: scale(1.18);} 30% {transform: scale(1);}
    45% {transform: scale(1.12);}}
@keyframes pc-crawl {from {left: 0;} to {left: calc(100% - 2rem);}}
@media (prefers-reduced-motion: reduce) {.pc-heart, .pc-baby {animation: none;}}
</style>
<div class="pc-header">
  <div class="pc-title">PregnaCare AI <span class="pc-heart">💗</span></div>
  <div class="pc-sub">Caring, evidence-based pregnancy guidance from WHO and MoHFW guidelines</div>
  <div class="pc-track"><span class="pc-baby">👶</span></div>
</div>
"""

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
        chunks.append({"text": text, "source": source, "file": meta["source"], "page": meta["page"],
                       "similarity": 1 - distance})
    return chunks


def source_link(c):
    # markdown link that opens the PDF at the first page of the chunk (pages look like "68" or "68-69")
    first_page = str(c["page"]).split("-")[0]
    return f"[{c['source']}, page {c['page']}](app/static/{quote(c['file'])}#page={first_page})"


def best_score(chunks):
    # the threshold is checked against this: cosine similarity between the question and the best chunk
    return max(c["similarity"] for c in chunks)


def ask_llm(question, trimester, chunks):
    # returns the answer text and the token usage reported by Groq
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
    return response.choices[0].message.content, response.usage


def translate(text, language):
    # one short Groq call; keeps the markdown (**bold**, bullet points) and numbers
    response = Groq().chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": f"Translate the user's text into {language}. Keep the meaning exact. "
                                          "Keep markdown like **bold** and bullet points, and keep numbers. "
                                          "Reply with the translation only."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content, response.usage


def transcribe(audio_bytes, lang):
    # speech to text with Groq Whisper (same API key); the language hint helps with Hindi and Bengali
    result = Groq().audio.transcriptions.create(file=("question.wav", audio_bytes),
                                                model="whisper-large-v3-turbo", language=lang)
    return result.text.strip()


def speak(text, lang):
    # reads the answer aloud with gTTS (needs internet); returns mp3 bytes
    from gtts import gTTS
    text = text.split("**Sources:**")[0].split("\n---\n")[0]  # skip the source links and the repeated English part
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text).replace("**", "").replace("🚨", "")
    audio = io.BytesIO()
    gTTS(text, lang=lang).write_to_fp(audio)
    return audio.getvalue()


def fixed_message(key, english, lang):
    # safety messages use fixed translations; the 108 message is shown in both languages
    if lang == "en":
        return english
    local = LOCAL_MESSAGES[lang][key]
    return f"{local}\n\n---\n\n{english}" if key == "escalation" else local


def format_sources(chunks):
    sources = dict.fromkeys(f"- {source_link(c)}" for c in chunks)  # removes duplicates
    return "**Sources:**\n" + "\n".join(sources)


def answer_sentences(answer):
    # sentences of the answer, leaving out the "When to see a doctor" part and the markdown
    text = answer.split("**When to see a doctor:**")[0].replace("**Answer:**", "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.replace("**", ""))
    return [p.strip(" -•") for p in parts if len(p.split()) >= 3]


def grounding_check(answer, chunks):
    # no extra LLM call: compare each answer sentence with the 4 chunks using the same MiniLM model
    sentences = answer_sentences(answer)
    if not sentences:
        return 100.0, []
    model = load_resources()[0]
    sentence_vectors = model.encode(sentences, normalize_embeddings=True)
    chunk_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    best = (sentence_vectors @ chunk_vectors.T).max(axis=1)  # best cosine similarity per sentence
    unsupported = [(s, float(b)) for s, b in zip(sentences, best) if b < GROUNDING_CUTOFF]
    return 100 * (len(sentences) - len(unsupported)) / len(sentences), unsupported


def confidence_score(best_similarity, grounding):
    # confidence = 50% "did we find good evidence" + 50% "does the answer stick to it"
    #   retrieval part: 0 at the refusal threshold (0.40), 100 at a strong match (0.70 or more), linear in between
    #   grounding part: the grounding % from grounding_check
    # label: High >= 75, Medium >= 50, else Low; and always Low if under half of the sentences are supported
    retrieval = min(max((best_similarity - SIMILARITY_THRESHOLD) / (0.70 - SIMILARITY_THRESHOLD), 0), 1) * 100
    score = 0.5 * retrieval + 0.5 * grounding
    if score >= 75 and grounding >= 50:
        return score, "High"
    return score, ("Medium" if score >= 50 and grounding >= 50 else "Low")


def find_evidence(question, sidebar_week):
    # decides which path runs (red_flag / refused / answer); evaluate.py uses this too
    if is_red_flag(question):
        return "red_flag", None, []  # safety check comes first, nothing else runs

    week = get_week(question) or sidebar_week
    trimester = get_trimester(week) if week else None
    chunks = retrieve(question, trimester)
    path = "refused" if best_score(chunks) < SIMILARITY_THRESHOLD else "answer"
    return path, trimester, chunks


def get_response(question, sidebar_week, lang="en"):
    # returns the assistant message: answer, evidence chunks, a caption saying which path ran, and metrics
    # Hindi/Bengali: red flag on the original text -> translate to English -> normal English flow
    #                (red flag again, retrieval, answer, validator) -> translate the checked answer back
    start = time.perf_counter()
    metrics = {}
    message = {"role": "assistant", "chunks": [], "metrics": metrics, "lang": lang}

    def finish(content, caption):
        metrics["total_s"] = time.perf_counter() - start
        return {**message, "content": content, "caption": caption}

    def add_translation(usage, seconds):
        metrics["translate_s"] = metrics.get("translate_s", 0) + seconds
        metrics["translate_tokens"] = metrics.get("translate_tokens", 0) + usage.total_tokens

    english = question
    if lang != "en":
        if is_red_flag(question):  # danger words in Hindi/Bengali/Hinglish, before any AI call
            metrics["retrieval_ms"] = (time.perf_counter() - start) * 1000
            return finish(fixed_message("escalation", ESCALATION_MESSAGE, lang), "Red-flag escalation")
        t = time.perf_counter()
        try:
            english, usage = translate(question, "English")
        except Exception as e:
            return finish(f"Could not reach the AI service, please try again. ({e})", "Translation failed")
        add_translation(usage, time.perf_counter() - t)
        message["english_question"] = english

    t = time.perf_counter()
    path, trimester, chunks = find_evidence(english, sidebar_week)  # runs the red-flag check on the English text
    metrics["retrieval_ms"] = (time.perf_counter() - t) * 1000
    message["chunks"] = chunks

    if path == "red_flag":
        return finish(fixed_message("escalation", ESCALATION_MESSAGE, lang), "Red-flag escalation")
    if path == "refused":
        return finish(fixed_message("not_enough", NOT_ENOUGH_INFO, lang),
                      f"Refused: not enough evidence (best score {best_score(chunks):.2f})")

    t = time.perf_counter()
    try:
        answer, usage = ask_llm(english, trimester, chunks)
    except Exception as e:
        return finish(f"Could not reach the AI service, please try again. ({e})", "LLM call failed")
    metrics["llm_s"] = time.perf_counter() - t
    metrics["tokens"] = (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)

    if not answer:
        return finish("I could not generate an answer, please try again.", "LLM returned no answer")
    if not is_safe_answer(answer):  # always checked on the English answer
        return finish(fixed_message("safe", SAFE_MESSAGE, lang), "Answer blocked by safety validation")

    grounding, unsupported = grounding_check(answer, chunks)  # scores also use the English answer
    confidence, level = confidence_score(best_score(chunks), grounding)
    metrics.update(grounding=grounding, hallucination=100 - grounding, unsupported=unsupported,
                   confidence=confidence, level=level)
    caption = f"Answered from {len(chunks)} evidence chunks"

    if lang != "en":
        message["english_answer"] = answer
        t = time.perf_counter()
        try:
            answer, usage = translate(answer, LANGUAGE_NAMES[lang])
            add_translation(usage, time.perf_counter() - t)
            caption += f" (translated to {LANGUAGE_NAMES[lang]})"
        except Exception:
            caption += " (translation failed, showing English)"
    return finish(f"{answer}\n\n{format_sources(chunks)}", caption)


def pregnancy_from_lmp(lmp, today):
    # Naegele's rule: due date = first day of last period (LMP) + 280 days (40 weeks)
    days = (today - lmp).days
    if days < 0:
        return None, "future"
    if days > 42 * 7:
        return None, "too_old"
    info = {"days_pregnant": days, "week": days // 7, "days": days % 7, "due": lmp + timedelta(days=280)}
    info["trimester"] = get_trimester(info["week"])
    return info, None


DATE_ERRORS = {
    ("lmp", "future"): "That date is in the future 🙂 Please pick the first day of your last period.",
    ("lmp", "too_old"): "That is more than 42 weeks ago. Please check the date, or ask your doctor.",
    ("due", "future"): "That due date is more than 40 weeks away. Please check the date.",
    ("due", "too_old"): "That due date was more than 2 weeks ago. Please check the date, or ask your doctor.",
}


def due_date_sidebar():
    # sidebar card; returns the current pregnancy week for retrieval (or None)
    st.sidebar.subheader("🗓️ Expected due date")
    mode = st.sidebar.radio("I know", ["First day of last period (LMP)", "I know my due date"])
    if mode == "I know my due date":
        due = st.sidebar.date_input("Due date", value=None, format="DD/MM/YYYY")
        lmp, kind = (due - timedelta(days=280) if due else None), "due"
    else:
        lmp = st.sidebar.date_input("First day of last period (LMP)", value=None, format="DD/MM/YYYY")
        kind = "lmp"
    if not lmp:
        st.sidebar.caption("Pick a date to see your due date and week.")
        return None

    info, error = pregnancy_from_lmp(lmp, date.today())
    if error:
        st.sidebar.warning(DATE_ERRORS[(kind, error)])
        return None

    to_go = 280 - info["days_pregnant"]
    st.sidebar.markdown(f"**Due date:** {info['due']:%d %B %Y}")
    st.sidebar.markdown(f"**You are:** {info['week']} weeks {info['days']} days ({info['trimester']} trimester)")
    st.sidebar.markdown(f"**Days to go:** {to_go}" if to_go >= 0 else f"**Past due date by** {-to_go} days")
    st.sidebar.progress(min(info["days_pregnant"] / 280, 1.0), text=f"{info['days_pregnant'] * 100 // 280}% of 40 weeks")
    st.sidebar.caption("Estimated using Naegele's rule; your doctor's ultrasound date is more accurate.")
    return info["week"] or None


def show_metrics(message):
    m = message["metrics"]
    with st.expander("Answer metrics"):
        lines = [f"- Retrieval time: {m['retrieval_ms']:.0f} ms"] if "retrieval_ms" in m else []
        if "llm_s" in m:
            prompt, completion, total = m["tokens"]
            lines += [f"- LLM time: {m['llm_s']:.1f} s", f"- Tokens: {prompt} in + {completion} out = {total}"]
        if "translate_s" in m:
            lines.append(f"- Translation: {m['translate_s']:.1f} s, {m['translate_tokens']} tokens")
        lines.append(f"- Total time: {m['total_s']:.1f} s")
        if "grounding" in m:
            lines += [f"- Grounding: {m['grounding']:.0f}% of sentences supported by the evidence",
                      f"- Hallucination score: {m['hallucination']:.0f}",
                      f"- Confidence: {m['confidence']:.0f}% ({m['level']})"]
        st.markdown("\n".join(lines))
        if m.get("unsupported"):
            st.markdown("**Sentences not clearly supported by the evidence:**")
            for sentence, score in m["unsupported"]:
                st.markdown(f"- {sentence} *(best match {score:.2f})*")
        if message.get("english_question"):
            st.markdown(f"**Question in English:** {message['english_question']}")
        if message.get("english_answer"):
            st.markdown("**English answer (the scores are computed on this):**")
            st.markdown(message["english_answer"])


def show_message(message, i):
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        if message.get("caption") == "Red-flag escalation":
            st.error(message["content"])  # the emergency message stays in a plain red box
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant" and st.button("🔊 Listen", key=f"listen_{i}"):
            try:
                st.audio(speak(message["content"], message.get("lang", "en")), format="audio/mp3", autoplay=True)
            except Exception:
                st.warning("Sorry, reading aloud is not available right now (it needs internet). "
                           "The answer is written above. 🙏")
        level = message.get("metrics", {}).get("level")
        if level:
            st.badge(f"Confidence {message['metrics']['confidence']:.0f}% · {level}", color=BADGE_COLORS[level])
        if message.get("caption"):
            st.caption(message["caption"])
        if message.get("chunks"):
            with st.expander("Retrieved evidence"):
                for c in message["chunks"]:
                    st.markdown(f"**{source_link(c)}** (similarity {c['similarity']:.2f})")
                    st.text(c["text"])
        if message.get("metrics"):
            show_metrics(message)


def performance_tab():
    import altair as alt
    import pandas as pd

    st.subheader("Retrieval and safety: baseline vs final")
    results = pd.read_csv("results.csv")
    names = {"red_flag_recall": "Red-flag recall", "false_alarm_rate": "False alarms (lower is better)",
             "hit_rate_at_1": "Hit@1", "hit_rate_at_4": "Hit@4", "answerable_answered": "Answerable answered",
             "out_of_scope_refused": "Out-of-scope refused", "week_accuracy": "Week accuracy"}
    pair = results[results["config"].isin(["baseline", "final"])].drop(columns="threshold")
    pair = pair.melt(id_vars="config", var_name="metric", value_name="percent").replace({"metric": names})
    chart = alt.Chart(pair).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("metric:N", title=None, sort=list(names.values()), axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("config:N", sort=["baseline", "final"]),
        x=alt.X("percent:Q", title="%", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(tickCount=5)),
        color=alt.Color("config:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["baseline", "final"], range=["#2a78d6", "#eb6834"])),
        tooltip=["config", "metric", "percent"],
    )
    st.altair_chart(chart, width="stretch")
    st.caption("Higher is better for every metric except false_alarm_rate. All configs:")
    st.dataframe(results, hide_index=True)

    st.subheader("Answer quality (real LLM, 10 questions)")
    if not os.path.exists("answer_results.csv"):
        st.info("Run `python evaluate.py --llm` to create answer_results.csv.")
        return
    answers = pd.read_csv("answer_results.csv")
    answered = answers.dropna(subset=["grounding"])
    cols = st.columns(4)
    cols[0].metric("Avg grounding", f"{answered['grounding'].mean():.0f}%")
    cols[1].metric("Avg hallucination", f"{answered['hallucination'].mean():.0f}")
    cols[2].metric("Keyword in answer", f"{answers['keyword_found'].mean() * 100:.0f}%")
    cols[3].metric("Avg latency", f"{answers['total_s'].mean():.1f} s")
    st.dataframe(answers, hide_index=True)


# streamlit runs this file as __main__; the check lets evaluate.py import the functions above
if __name__ == "__main__":
    st.set_page_config(page_title="PregnaCare AI", page_icon="🤰")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.warning(
        "This chatbot gives general information from WHO and government antenatal guidelines. "
        "It is not a doctor and does not replace medical advice. In an emergency, call 108."
    )
    lang = LANGUAGES[st.sidebar.selectbox("🌐 Language / भाषा / ভাষা", list(LANGUAGES))]
    sidebar_week = due_date_sidebar()  # a week written in the question still wins (see find_evidence)

    chat_tab, performance = st.tabs(["💬 Chat", "📊 Performance"])
    with performance:
        performance_tab()

    with chat_tab:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        with st.chat_message("assistant", avatar="🌸"):
            welcome = "Hello mom-to-be 💕 Ask me anything about your pregnancy."
            st.markdown(welcome if lang == "en" else LOCAL_MESSAGES[lang]["welcome"])
        history = st.container()  # drawn above the inputs, so new messages appear above them too
        # voice question: only transcribe a recording once (streamlit reruns the script on every click)
        voice = st.audio_input("🎤 Or ask by voice")
        question = st.chat_input("Ask a question about your pregnancy")
        if voice is not None and hash(voice.getvalue()) != st.session_state.get("last_voice"):
            st.session_state.last_voice = hash(voice.getvalue())
            try:
                question = transcribe(voice.getvalue(), lang)
            except Exception:
                st.warning("Sorry, I couldn't hear that clearly. Please try again, or type your question. 🙏")

        with history:
            for i, message in enumerate(st.session_state.messages):
                show_message(message, i)
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                show_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)

                with st.spinner("Checking the guidelines..."):
                    message = get_response(question, sidebar_week, lang)
                st.session_state.messages.append(message)
                show_message(message, len(st.session_state.messages) - 1)
