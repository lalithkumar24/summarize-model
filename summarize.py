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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure NLTK data path using current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
nltk.data.path.append(os.path.join(current_dir, "nltk_data"))

# Download required NLTK resources
try:
    nltk.download("punkt")
    nltk.download("stopwords")
except Exception as e:
    logger.error(f"Failed to download NLTK resources: {e}")


class MultilingualSummarizer:
    def __init__(self):
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
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
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
            translated_text = (
                text if source_lang == "en" else self.translate_text(text, "en")
            )

            # Tokenize sentences
            sentences = sent_tokenize(translated_text)
            if len(sentences) <= num_sentences:
                return {
                    "original_text": text,
                    "source_language": source_lang,
                    "summary": translated_text,
                }

            # Preprocess words
            stop_words = self.get_stopwords(source_lang)
            words = [
                word
                for word in word_tokenize(translated_text.lower())
                if word.isalnum() and word not in stop_words
            ]

            # Compute word frequencies
            word_frequencies = FreqDist(words)

            # Score sentences
            sentence_scores = {
                sentence: sum(
                    word_frequencies.get(word, 0)
                    for word in word_tokenize(sentence.lower())
                )
                for sentence in sentences
            }

            # Select top sentences
            summary = " ".join(
                nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            )

            # Translate summary back if needed
            if source_lang != "en":
                summary = self.translate_text(summary, source_lang)

            return {
                "original_text": text,
                "source_language": source_lang,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {"error": str(e), "status": "failure"}


# Flask Application
app = Flask(__name__)
summarizer = MultilingualSummarizer()


@app.route("/", methods=["POST"])
def handle_request():
    """Unified endpoint for OpenServ"""
    try:
        data = request.get_json()

        if not data or "workspace" not in data or "goal" not in data["workspace"]:
            return (
                jsonify({"error": "Invalid request format", "status": "failure"}),
                400,
            )

        goal = data["workspace"]["goal"]
        task, extracted_text = parse_goal(goal)

        if not extracted_text:
            return (
                jsonify({"error": "No valid text found in goal", "status": "failure"}),
                400,
            )

        if task == "summarize":
            result = summarizer.summarize(extracted_text)
        elif task == "translate":
            result = handle_translation(extracted_text, data)
        else:
            return jsonify({"error": "Unsupported task", "status": "failure"}), 400

        return jsonify(
            {
                "type": "respond-chat-message",
                "me": data["me"],
                "messages": [{"author": "Summarizer", "message": result}],
            }
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e), "status": "failure"}), 500


def parse_goal(goal):
    """Extracts the task and text from workspace.goal"""
    if "summary" in goal.lower() or "summarize" in goal.lower():
        task = "summarize"
    elif "translate" in goal.lower():
        task = "translate"
    else:
        return None, None  # No valid task detected

    # Extract quoted text if available
    start = goal.find(":")
    if start != -1:
        extracted_text = goal[start + 1 :].strip()
        return task, extracted_text
    return None, None


def handle_translation(text, data):
    """Handles translation using OpenServ's expected format"""
    try:
        curr_lang = data.get("curr_lang", "en")
        target_lang = data.get("target_lang", "en")

        if not curr_lang or not target_lang:
            return {"error": "Missing language fields", "status": "failure"}

        detected_lang = summarizer.detect_language(text)
        if detected_lang != curr_lang:
            return {
                "error": f"Detected language '{detected_lang}' does not match provided '{curr_lang}'",
                "status": "failure",
            }

        translated_text = summarizer.translate_text(text, target_lang)
        return {
            "original_text": text,
            "translated_text": translated_text,
            "source_language": curr_lang,
            "target_language": target_lang,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return {"error": str(e), "status": "failure"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
