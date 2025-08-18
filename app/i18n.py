from langdetect import detect, LangDetectException

def detect_lang(text):
    try:
        code = detect(text)
        return "es" if code.startswith("es") else "en"
    except LangDetectException:
        return "en"
