import os
import logging
from flask import Flask, request, jsonify
from transformers import pipeline
from googletrans import Translator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Hugging Face summarization model
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()


class NLPProcessor:
    """Handles summarization and translation tasks"""

    def summarize(self, text, min_length=50, max_length=150):
        """Summarizes a given text using a Transformer model"""
        try:
            logger.info("Summarizing text...")
            summary = summarization_pipeline(
                text, min_length=min_length, max_length=max_length
            )
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return None

    def translate(self, text, target_lang="en"):
        """Translates text to the target language"""
        try:
            return translator.translate(text, dest=target_lang).text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None


app = Flask(__name__)
nlp_processor = NLPProcessor()


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
            result = nlp_processor.summarize(extracted_text)
        elif task == "translate":
            target_lang = data.get("target_lang", "en")
            result = nlp_processor.translate(extracted_text, target_lang)
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
        return None, None

    start = goal.find(":")
    if start != -1:
        extracted_text = goal[start + 1 :].strip()
        return task, extracted_text
    return None, None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
