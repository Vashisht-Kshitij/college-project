# Safety checks and pregnancy week helpers used by app.py
import re
import unicodedata

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

# Hindi and Bengali danger words: Devanagari, Bengali script and common romanized (Hinglish) forms.
# plain substrings, because \b does not work well with Indian scripts. "khoon ki jaanch" (blood test)
# and "khoon ki kami" (anaemia) are routine questions, so they are left out on purpose.
LOCAL_RED_FLAGS = [unicodedata.normalize("NFD", p) for p in [
    # Hindi script
    r"खून(?! की (?:जांच|जाँच|कमी))", r"रक्तस्राव", r"ब्लीडिंग",
    r"(?:हिल|हिलना|हिलता|हिलती)\s*(?:नहीं|बंद)", r"नहीं\s+हिल", r"हलचल\s*(?:नहीं|कम|बंद)",
    r"(?:चेहर|हाथ)\S*\s+(?:\S+\s+){0,2}सू?ज", r"सूजन",
    r"(?:तेज़?|बहुत|भयंकर)\s+सिर\s*दर्द", r"सिर\s*(?:में)?\s*(?:बहुत|तेज़?)\s+दर्द", r"धुंधल",
    r"दौरा|झटक|मिर्गी|बेहोश", r"(?:तेज़?|बहुत)\s+बुखार", r"पानी\s+(?:\S+\s+){0,2}(?:निकल|गिर|बह|टूट)",
    r"पेट\s+(?:में\s+)?(?:बहुत\s+)?(?:तेज़?|भयंकर|असहनीय)\s+दर्द", r"पेट\s+के\s+बल\s+गिर",
    r"सा(?:ं|ँ)स\s+(?:\S+\s+){0,3}(?:तकलीफ|दिक्कत|नहीं)",
    # Bengali script
    r"রক্তপাত", r"রক্ত\s*(?:পড়|যাচ্ছে|বের)", r"নড়ছে\s*না", r"নড়াচড়া\s*(?:কম|বন্ধ|নেই|করছে না)",
    r"(?:মুখ|হাত)\S*\s+(?:\S+\s+){0,2}ফুল", r"(?:প্রচণ্ড|খুব|তীব্র)\s+মাথা\s*ব্যথা", r"ঝাপসা",
    r"খিঁচুনি|অজ্ঞান|জ্ঞান হারা", r"(?:প্রচণ্ড|খুব|তীব্র)\s+জ্বর", r"(?:জল|পানি)\s+(?:\S+\s+){0,2}(?:ভাঙ|ভেঙ|বের|পড়|ঝর)",
    r"পেটে\s+(?:\S+\s+)?(?:প্রচণ্ড|তীব্র|অসহ্য)\s+ব্যথা", r"শ্বাস\s+(?:\S+\s+){0,3}(?:কষ্ট|পারছি না)",
    # romanized Hindi / Bengali
    r"\bkh?oo?n\b(?! ki (?:jaa?nch|kami|test))", r"bleeding ho",
    r"\b(?:bachh?a|bachch?a|bacch?a|baby)\s+(?:\w+\s+){0,2}(?:hil|hal)\w*\s+(?:nahi|nhi|nahin)\b", r"\b(?:nahi|nhi)\s+hil",
    r"\b(?:chehr[ae]|haath)\w*\s+(?:\w+\s+){0,2}s[ou]+j", r"\bsoojan\b",
    r"\b(?:tez|bahut|bhayankar)\s+sir\s*dard", r"\bdhundh?la\b", r"\b(?:daura|jhatke|mirgi|behosh)\b",
    r"\b(?:tez|bahut)\s+bukhar", r"\bpaa?ni\s+(?:\w+\s+){0,2}(?:nikal|gir|beh|toot|tut)",
    r"\bpet\s+(?:me|mein|main)?\s*(?:bahut\s+)?(?:tez|bhayankar|zyada)\s+dard", r"\bsaa?ns\s+(?:\w+\s+){0,3}(?:taklee?f|dikkat|nahi)",
    r"\brokto\s+(?:por|jacch|ber)", r"\bnorche\s+na\b", r"\bmukh\s+phul", r"\b(?:ojnan|khichuni)\b",
]]

# fixed translations, so these safety messages never depend on the AI
LOCAL_MESSAGES = {
    "hi": {
        "escalation": "🚨 **यह खतरे का संकेत हो सकता है। कृपया इंतज़ार न करें।**\n\n- तुरंत अपने डॉक्टर से संपर्क करें\n"
                      "- या नज़दीकी अस्पताल जाएँ\n- या एम्बुलेंस के लिए **108** पर कॉल करें",
        "not_enough": "मेरे ज्ञान-आधार में इस बारे में पर्याप्त सत्यापित जानकारी नहीं है। कृपया अपने डॉक्टर से सलाह लें।",
        "safe": "मैं दवा की खुराक या निदान नहीं बता सकती। कृपया इस बारे में अपने डॉक्टर से सलाह लें।",
        "welcome": "नमस्ते, होने वाली माँ 💕 अपनी गर्भावस्था के बारे में मुझसे कुछ भी पूछें।",
    },
    "bn": {
        "escalation": "🚨 **এটি বিপদের লক্ষণ হতে পারে। দয়া করে অপেক্ষা করবেন না।**\n\n- এখনই আপনার ডাক্তারের সাথে যোগাযোগ করুন\n"
                      "- অথবা নিকটতম হাসপাতালে যান\n- অথবা অ্যাম্বুলেন্সের জন্য **108** নম্বরে কল করুন",
        "not_enough": "আমার জ্ঞানভান্ডারে এ বিষয়ে যথেষ্ট যাচাই করা তথ্য নেই। দয়া করে আপনার ডাক্তারের পরামর্শ নিন।",
        "safe": "আমি ওষুধের ডোজ বা রোগ নির্ণয় বলতে পারি না। দয়া করে এ বিষয়ে আপনার ডাক্তারের পরামর্শ নিন।",
        "welcome": "নমস্কার, হবু মা 💕 আপনার গর্ভাবস্থা নিয়ে আমাকে যা খুশি জিজ্ঞাসা করুন।",
    },
}

DOSE_PATTERNS = [
    r"\d\s*(mg|mcg|µg|gm?|iu)\b|\b(mg|mcg)\s*\d",
    r"(\d|\bone|\btwo|\bthree)\s*(ifa |iron |calcium |folic acid )?(tablets?|capsules?)\b",
]
DIAGNOSIS_PATTERN = r"\byou (may |might |probably )?have\b|\byou are suffering from\b|\byou're suffering from\b"
# "see a doctor if you have..." is normal advice, not a diagnosis
CONDITIONAL_YOU_HAVE = r"\b(if|when|whenever|unless|in case|whether|do)\s+you have\b"


def is_red_flag(question):
    # NFD so Indian-script letters match however they were typed (e.g. letters with a nukta dot)
    text = unicodedata.normalize("NFD", question.lower().replace("’", "'"))
    return any(re.search(pattern, text) for pattern in RED_FLAGS + LOCAL_RED_FLAGS)


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
