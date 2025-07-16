from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os, requests, cohere

# ──────────────────────────────────────────
# Flask + DB
# ──────────────────────────────────────────
app = Flask(__name__, static_folder='../frontend', static_url_path='/frontend')
app.secret_key = 'supersecretkey'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "hufflestudy.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ──────────────────────────────────────────
# Models
# ──────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# ──────────────────────────────────────────
# Routes: Auth
# ──────────────────────────────────────────
@app.route('/')
def home():
    return redirect('/frontend/index.html')

@app.route('/register', methods=['POST'])
def register():
    if User.query.filter_by(email=request.form['email']).first():
        flash('Email already registered.')
        return redirect('/frontend/register.html')
    db.session.add(User(
        name=request.form['name'],
        email=request.form['email'],
        password=generate_password_hash(request.form['password'])
    ))
    db.session.commit()
    flash('Registration successful!')
    return redirect('/frontend/login.html')

@app.route('/login', methods=['POST'])
def login():
    user = User.query.filter_by(email=request.form['email']).first()
    if user and check_password_hash(user.password, request.form['password']):
        return redirect('/frontend/dashboard.html')
    flash('Invalid credentials.')
    return redirect('/frontend/login.html')

# ──────────────────────────────────────────
# HuggingFace Summarizer
# ──────────────────────────────────────────
HF_MODEL_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN", "")

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.get_json(force=True).get('text', '').strip()
    if not text:
        return jsonify({'summary': '❗ No text provided.'}), 400

    try:
        resp = requests.post(
            HF_MODEL_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": text,
                "parameters": {
                    "max_length": 220,
                    "min_length": 80,
                    "num_beams": 4,
                    "length_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                    "do_sample": False
                }
            },
            timeout=60
        )
        resp.raise_for_status()
        return jsonify({'summary': resp.json()[0]['summary_text']}), 200
    except Exception as e:
        return jsonify({'summary': f'⚠️ HF error: {str(e)}'}), 500

# ──────────────────────────────────────────
# Cohere Quiz Generator
# ──────────────────────────────────────────
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
co = cohere.Client(COHERE_API_KEY)

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    topic = request.get_json(force=True).get('topic', '').strip()
    if not topic:
        return jsonify({'quiz': '❗ No topic provided.'}), 400

    prompt = (f"Generate 5 multiple-choice questions with 4 options each and mark the correct answer "
              f"for the topic: {topic}.\n"
              "Format:\n"
              "1. Question?\n"
              "   a. Option A\n"
              "   b. Option B\n"
              "   c. Option C\n"
              "   d. Option D\n"
              "Answer: a")

    try:
        resp = co.chat(model='command-r', message=prompt, temperature=0.7, max_tokens=500)
        return jsonify({'quiz': resp.text.strip()}), 200
    except Exception as e:
        return jsonify({'quiz': f'❌ Cohere error: {str(e)}'}), 500

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))
@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({'answer': '❗ Please enter a question.'}), 400

    try:
        resp = co.chat(
            model="command-r",  # or "command-r-plus"
            message=question,
            temperature=0.7,
            max_tokens=300
        )
        return jsonify({'answer': resp.text.strip()}), 200
    except Exception as e:
        return jsonify({'answer': f'⚠️ Cohere error: {str(e)}'}), 500


# ──────────────────────────────────────────
# Run App
# ──────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    print("🚀 HuffleStudy running → http://127.0.0.1:5000")
    app.run(debug=True)
