# AI Text Summarizer Backend (app.py)
# To run this:
# 1. Make sure you have Python installed.
# 2. Install Flask and spacy:
#    pip install Flask flask-cors
#    pip install spacy
# 3. Download the spacy model:
#    python -m spacy download en_core_web_sm
# 4. Run the server from your terminal:
#    python app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    nlp = None

def summarize_text(text, num_sentences=3):
    """
    Summarizes the input text using spaCy.
    """
    if nlp is None:
        return "SpaCy model is not loaded. Cannot summarize."

    doc = nlp(text)

    # 1. Get keywords (non-stopword tokens)
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    # 2. Calculate word frequencies
    word_freq = Counter(keywords)
    max_freq = max(word_freq.values(), default=1)
    
    # 3. Normalize frequencies
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_freq)
        
    # 4. Score sentences
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if sent in sentence_scores.keys():
                    sentence_scores[sent] += word_freq[word.text.lower()]
                else:
                    sentence_scores[sent] = word_freq[word.text.lower()]

    # 5. Get the top N sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # 6. Join them to form the summary
    summary = [sent.text for sent in summarized_sentences[:num_sentences]]
    
    return " ".join(summary)


@app.route('/summarize', methods=['POST'])
def summarize():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = summarize_text(text)
    
    return jsonify({"summary": summary})

if __name__ == '__main__':
    print("Starting Flask server for AI Summarizer...")
    print("Open summarizer.html in your browser to use.")
    app.run(debug=True, port=5000)
