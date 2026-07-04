from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load 3-class sentiment model directly with pipeline
MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
sentiment_analyzer = pipeline("sentiment-analysis", model=MODEL_NAME)

# Map labels to human-readable format
label_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Get prediction
    result = sentiment_analyzer(text)[0]
    sentiment = label_map.get(result['label'], result['label']).lower()
    score = float(result['score'])

    return jsonify({'sentiment': sentiment, 'score': score})

if __name__ == "__main__":
    # Run server on 0.0.0.0 so it's accessible in deployment
    app.run(host="0.0.0.0", port=5000, debug=True)
