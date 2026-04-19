from deep_translator import GoogleTranslator

def translate_batch(texts, target_lang):
    translated = []

    for t in texts:
        try:
            result = GoogleTranslator(source='auto', target=target_lang).translate(str(t))
            
            # fallback check
            if result is None or result.strip() == "":
                translated.append(t)
            else:
                translated.append(result)

        except Exception as e:
            translated.append(t)

    return translated
