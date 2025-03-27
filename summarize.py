import os
import nltk
from flask import Flask, request, jsonify
from googletrans import Translator
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
nltk.data.path.append("/app/nltk_data")  # Use pre-downloaded resources


class MultilingualSummarizer:
    def __init__(self):
        """Initialize the multilingual summarizer with a translator"""
        self.translator = Translator()

    def detect_language(self, text):
        """Detect language of the input text"""
        try:
            return self.translator.detect(text).lang
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return "en"  # Default to English

    def translate_text(self, text, target_lang="en"):
        """Translate text to target language"""
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails

    def get_stopwords(self, lang="english"):
        """Retrieve stopwords for a given language"""
        lang_map = {
            "en": "english", "es": "spanish", "fr": "french",
            "de": "german", "it": "italian", "pt": "portuguese", "ru": "russian"
        }
        try:
            return set(stopwords.words(lang_map.get(lang, "english")))
        except Exception as e:
            logger.error(f"Stopwords error for {lang}: {e}")
            return set()

    def summarize(self, text, num_sentences=3, target_lang="en"):
        """Generate a summary of multilingual text"""
        try:
            if not text or not isinstance(text, str):
                return {"error": "Invalid input text", "status": "failure"}

            # Detect source language
            source_lang = self.detect_language(text)
            logger.info(f"Detected language: {source_lang}")

            # Translate text to English for processing if needed
            translated_text = text if source_lang == "en" else self.translate_text(text, "en")

            # Tokenize sentences
            sentences = sent_tokenize(translated_text)
            if len(sentences) <= num_sentences:
                return {"original_text": text, "source_language": source_lang, "summary": translated_text}

            # Preprocess words
            stop_words = self.get_stopwords(source_lang)
            words = [word for word in word_tokenize(translated_text.lower()) if word.isalnum() and word not in stop_words]

            # Compute word frequencies
            word_frequencies = FreqDist(words)

            # Score sentences
            sentence_scores = {sentence: sum(word_frequencies.get(word, 0) for word in word_tokenize(sentence.lower())) for sentence in sentences}

            # Select top sentences
            summary = " ".join(nlargest(num_sentences, sentence_scores, key=sentence_scores.get))

            # Translate summary back if needed
            if source_lang != "en":
                summary = self.translate_text(summary, source_lang)

            return {"original_text": text, "source_language": source_lang, "summary": summary}

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"error": str(e), "status": "failure"}


# Flask Application
app = Flask(__name__)
summarizer = MultilingualSummarizer()


@app.route("/summarize", methods=["POST"])
def summarize_text():
    """API endpoint for text summarization"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided", "status": "failure"}), 400

        result = summarizer.summarize(data["text"], data.get("num_sentences", 3), data.get("target_language", "en"))
        return jsonify(result)

    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        return jsonify({"error": str(e), "status": "failure"}), 500


@app.route("/translate", methods=["POST"])
def translate_text():
    """API endpoint for text translation"""
    try:
        data = request.get_json()
        if not data or "text" not in data or "target_language" not in data:
            return jsonify({"error": "Invalid input", "status": "failure"}), 400

        translated_text = summarizer.translate_text(data["text"], data["target_language"])
        return jsonify({"original_text": data["text"], "translated_text": translated_text, "target_language": data["target_language"]})

    except Exception as e:
        logger.error(f"Translation API error: {e}")
        return jsonify({"error": str(e), "status": "failure"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Multilingual Summarizer", "version": "1.2.0"}), 200


@app.route("/", methods=["GET"])
def home():
    return "Multilingual Text Summarizer API is running!"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

