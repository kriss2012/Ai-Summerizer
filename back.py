from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spacy
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

    # 1. Get keywords (non-stopword, non-punctuation tokens)
    keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    # 2. Calculate word frequencies
    word_freq = Counter(keywords)
    if not word_freq:
        return "Not enough content to summarize."
        
    max_freq = max(word_freq.values())
    
    # 3. Normalize frequencies
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_freq)
        
    # 4. Score sentences based on word frequencies
    sentence_scores = {}
    for sent in doc.sents:
        # Ignore very short sentences
        if len(sent) < 5:
            continue
        for word in sent:
            if word.text.lower() in word_freq:
                if sent in sentence_scores:
                    sentence_scores[sent] += word_freq[word.text.lower()]
                else:
                    sentence_scores[sent] = word_freq[word.text.lower()]

    if not sentence_scores:
        return "Could not determine the main sentences. Please provide more text."

    # 5. Get the top N sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # 6. Join them to form the summary
    summary = [sent.text.strip() for sent in summarized_sentences[:num_sentences]]
    
    return " ".join(summary)

@app.route('/')
def index():
    """Serve the frontend HTML file."""
    return send_from_directory('.', 'App.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text', '')
    num_sentences = data.get('num_sentences', 3)

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if len(text.split()) < 20: # Basic check for minimum text length
        return jsonify({"error": "Please provide a longer text for a better summary."}), 400

    summary = summarize_text(text, num_sentences)
    
    return jsonify({"summary": summary})

if __name__ == '__main__':
    print("Starting Flask server for AI Summarizer...")
    print("Open http://127.0.0.1:5000 in your browser to use.")
    app.run(host='0.0.0.0', debug=True, port=5000)
