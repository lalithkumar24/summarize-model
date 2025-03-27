import os
import nltk
import requests
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
import logging

# Ensure NLTK resources are downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"NLTK download error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self):
        """
        Initialize the text summarizer
        """
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Preprocess the input text by tokenizing and removing stop words
        """
        try:
            words = word_tokenize(text.lower())
            words = [word for word in words if word.isalnum() and word not in self.stop_words]
            return words
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return []

    def summarize(self, text, num_sentences=3):
        """
        Generate a summary of the text
        """
        try:
            # Tokenize sentences
            sentences = sent_tokenize(text)
            
            # Handle short texts
            if len(sentences) <= num_sentences:
                return text
            
            # Preprocess and get word frequencies
            words = self.preprocess_text(text)
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
            
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "Unable to generate summary"

# Flask Application
app = Flask(__name__)
summarizer = TextSummarizer()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """
    Endpoint to receive text and generate summary
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
        
        # Generate summary
        summary = summarizer.summarize(text, num_sentences)
        
        # Log successful summarization
        logger.info(f"Successfully summarized text (Length: {len(text)})")
        
        # Return summary
        return jsonify({
            'original_text': text,
            'summary': summary,
            'status': 'success'
        })
    
    except Exception as e:
        # Log and return error
        logger.error(f"Summarization error: {e}")
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
        'service': 'NLTK Text Summarizer',
        'version': '1.0.0'
    }), 200

def create_app():
    """
    Create and configure Flask app
    """
    return app

if __name__ == '__main__':
    # Determine port from environment variable or default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
