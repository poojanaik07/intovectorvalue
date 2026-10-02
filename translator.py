from deep_translator import GoogleTranslator
import streamlit as st

def translate_batch(texts, target_lang, progress_callback=None):
    """
    Translate a list of texts to target_lang.
    Uses GoogleTranslator with chunking (max 4500 chars per chunk).
    Falls back to original text on error.
    """
    translated = []
    total = len(texts)

    for i, t in enumerate(texts):
        try:
            text_str = str(t).strip()

            # GoogleTranslator has a ~5000 char limit; truncate safely
            if len(text_str) > 4500:
                text_str = text_str[:4500]

            result = GoogleTranslator(source='auto', target=target_lang).translate(text_str)

            # fallback check
            if result is None or result.strip() == "":
                translated.append(t)
            else:
                translated.append(result)

        except Exception:
            translated.append(t)  # keep original on error

        if progress_callback:
            progress_callback(i + 1, total)

    return translated
