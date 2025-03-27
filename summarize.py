import os
import nltk
from flask import Flask, request, jsonify
from googletrans import Translator
import logging
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTK download error: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualSummarizer:
    def __init__(self):
        """
        Initialize multilingual summarizer with translator
        """
        self.translator = Translator()

    def detect_language(self, text):
        """
        Detect language of the input text
        """
        try:
            detection = self.translator.detect(text)
            return detection.lang
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'  # Default to English

    def translate_text(self, text, target_lang='en'):
        """
        Translate text to target language
        """
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def get_stopwords(self, lang='english'):
        """
        Get stopwords for different languages
        """
        try:
            # Mapping of language codes to NLTK stopwords
            lang_map = {
                'en': 'english',
                'es': 'spanish',
                'fr': 'french',
                'de': 'german',
                'it': 'italian',
                'pt': 'portuguese',
                'ru': 'russian'
            }
            
            # Use English stopwords as default
            nltk_lang = lang_map.get(lang, 'english')
            return set(stopwords.words(nltk_lang))
        except Exception as e:
            logger.error(f"Stopwords error for {lang}: {e}")
            return set()

    def summarize(self, text, num_sentences=3, target_lang='en'):
        """
        Generate summary for multilingual text
        """
        try:
            # Detect source language
            source_lang = self.detect_language(text)
            logger.info(f"Detected language: {source_lang}")

            # Translate to English for processing if not already English
            if source_lang != 'en':
                translated_text = self.translate_text(text, 'en')
            else:
                translated_text = text

            # Get appropriate stopwords
            stop_words = self.get_stopwords(source_lang)

            # Tokenize sentences and words
            sentences = sent_tokenize(translated_text)
            
            # Handle short texts
            if len(sentences) <= num_sentences:
                return translated_text

            # Preprocess text
            words = word_tokenize(translated_text.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words]

            # Calculate word frequencies
            word_frequencies = FreqDist(words)
            
            # Score sentences
            sentence_scores = {}
            for sentence in sentences:
                for word in word_tokenize(sentence.lower()):
                    if word in word_frequencies:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_frequencies[word]
                        else:
                            sentence_scores[sentence] += word_frequencies[word]
            
            # Get top sentences
            summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            summary = ' '.join(summary_sentences)

            # Translate summary back to source language if needed
            if source_lang != 'en':
                summary = self.translate_text(summary, source_lang)

            return {
                'original_text': text,
                'source_language': source_lang,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return {
                'error': str(e),
                'status': 'failure'
            }

# Flask Application
app = Flask(__name__)
summarizer = MultilingualSummarizer()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Endpoint to receive multilingual text and generate summary
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'status': 'failure'
            }), 400
        
        # Extract text and optional parameters
        text = data['text']
        num_sentences = data.get('num_sentences', 3)
        target_lang = data.get('target_language', 'en')
        
        # Generate summary
        result = summarizer.summarize(text, num_sentences, target_lang)
        
        # Return summary
        return jsonify(result)
    
    except Exception as e:
        # Return error
        return jsonify({
            'error': str(e),
            'status': 'failure'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Multilingual Text Summarizer',
        'version': '1.0.0'
    }), 200

def create_app():
    """
    Create and configure Flask app
    """
    return app

if __name__ == '__main__':
    # Determine port from environment variable or default
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
