# Safety checks and pregnancy week helpers used by app.py
import re

ESCALATION_MESSAGE = """🚨 **This could be a danger sign. Please do not wait.**

- Contact your doctor right away
- Or go to the nearest hospital
- Or call **108** for an ambulance

This chatbot cannot assess emergencies, so it has not generated an answer."""

SAFE_MESSAGE = "I can't give specific medicine doses or a diagnosis. Please consult your doctor about this."


def near(a, b):
    # a and b within 4 words of each other, in either order
    gap = r"\W+(?:\w+\W+){0,4}?"
    return rf"\b(?:{a})\b{gap}\b(?:{b})\b|\b(?:{b})\b{gap}\b(?:{a})\b"


# danger signs, matched as word combinations so different wordings still match
RED_FLAGS = [
    r"\b(bleed\w*|bled)\b",
    r"\bblood\b(?! (pressure|tests?|sugar|group|type|report|count|levels?|glucose))",  # these are routine questions
    near("severe|bad|terrible", "headaches?"),
    near("blurred|blurry", "vision|eyesight|eyes?"),
    r"convuls|seizure|concussion|unconscious|\bfaint(ed|ing)?\b",
    r"\bfits\b|\b(had|having|have|got) a fit\b(?! (pregnancy|body|baby|woman|mother))",  # not "stay fit" or "benefits"
    near("head", "injury|injured|hit"),
    near(r"fell(?! asleep)|fallen(?! asleep)|falling(?! asleep)|fall(?! asleep)", "stomach|belly|abdomen|tummy"),
    near("severe", "abdominal|abdomen|stomach|belly|tummy"),
    near("high", "fever|temperature"),
    near("baby", r"(not|no|stopped|\w+n't|isnt|hasnt|doesnt)\s+(\w+\s+){0,2}mov\w*|(can't|cannot|cant|don't|dont) feel"),
    r"\bno (fetal |foetal |baby |baby's )?movements?\b",  # not "no bowel movement"
    r"(reduced|less|decreased) (fetal |foetal |baby |baby's )?movement",
    r"waters? (has |have |is |are )?(broke|broken|breaking)",
    r"leaking (fluid|water)",
    near("swelling|swollen|puffy", "face|hands?|fingers?"),
    near("difficulty|trouble|problem", "breathing") + r"|(can't|cannot|cant) breathe|shortness of breath",
]

DOSE_PATTERNS = [
    r"\d\s*(mg|mcg|µg|gm?|iu)\b|\b(mg|mcg)\s*\d",
    r"(\d|\bone|\btwo|\bthree)\s*(ifa |iron |calcium |folic acid )?(tablets?|capsules?)\b",
]
DIAGNOSIS_PATTERN = r"\byou (may |might |probably )?have\b|\byou are suffering from\b|\byou're suffering from\b"
# "see a doctor if you have..." is normal advice, not a diagnosis
CONDITIONAL_YOU_HAVE = r"\b(if|when|whenever|unless|in case|whether|do)\s+you have\b"


def is_red_flag(question):
    text = question.lower().replace("’", "'")
    return any(re.search(pattern, text) for pattern in RED_FLAGS)


def is_safe_answer(answer):
    text = answer.lower().replace("’", "'")
    text = re.sub(CONDITIONAL_YOU_HAVE, "", text)
    has_dose = any(re.search(pattern, text) for pattern in DOSE_PATTERNS)
    return not has_dose and not re.search(DIAGNOSIS_PATTERN, text)


def get_week(question):
    match = re.search(r"(\d{1,2})\s*-?\s*(?:weeks?|wks?)\b|\bweek\s*(\d{1,2})\b", question.lower())
    if not match:
        return None
    week = int(match.group(1) or match.group(2))
    return week if 1 <= week <= 42 else None


def get_trimester(week):
    if week <= 12:
        return "first"
    return "second" if week <= 27 else "third"
