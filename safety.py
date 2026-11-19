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
    r"convuls|seizure|concussion|unconscious|\bfaint(ed|ing)?\b|passed out|blacked out",
    r"\bfits\b|\b(had|having|have|got) a fit\b(?! (pregnancy|body|baby|woman|mother))",  # not "stay fit" or "benefits"
    near("head", "injury|injured|hit"),
    near(r"fell(?! asleep)|fallen(?! asleep)|falling(?! asleep)|fall(?! asleep)", "stomach|belly|abdomen|tummy"),
    near("severe|unbearable|extreme|worst|terrible", "abdominal|abdomen|stomach|belly|tummy"),
    near("high", "fever|temperature"),
    near("baby", r"(not|no|stopped|\w+n't|isnt|hasnt|doesnt)\s+(\w+\s+){0,2}mov\w*|(can't|cannot|cant|don't|dont) feel"),
    r"\bno (fetal |foetal |baby |baby's )?movements?\b",  # not "no bowel movement"
    r"(reduced|less|decreased) (fetal |foetal |baby |baby's )?movement",
    r"waters? (has |have |is |are )?(broke|broken|breaking)",
    near("waters?|fluid", r"leak\w*"),
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


# the number must be right next to pregnancy words, so "cough for 2 weeks" is not read as week 2
WEEK_PATTERNS = [
    r"(\d{1,2})\s*-?\s*(?:weeks?|wks?)\s+(?:pregnant|along|into (?:my |the )?pregnancy|of (?:my |the )?pregnancy)",
    r"\bweek\s*(\d{1,2})\s+of\s+(?:my |the )?pregnancy",
    r"(\d{1,2})(?:st|nd|rd|th)\s+week\s+of\s+(?:my |the )?pregnancy",
    r"pregnant\s+(?:for\s+)?(\d{1,2})\s*(?:weeks?|wks?)",
]


def get_week(question):
    for pattern in WEEK_PATTERNS:
        match = re.search(pattern, question.lower())
        if match and 1 <= int(match.group(1)) <= 42:
            return int(match.group(1))
    return None


def get_trimester(week):
    if week <= 12:
        return "first"
    return "second" if week <= 27 else "third"
