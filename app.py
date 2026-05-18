from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session storage

# --- Load model and scaler ---
MODEL_PATH = "models/personality_model_10.pkl"
SCALER_PATH = "models/scaler_10.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --- Top 10 questions (must match order from train.py) ---
questions = [
    "I enjoy social gatherings and meeting new people.",
    "I prefer making detailed plans before acting.",
    "I rely on my imagination to solve problems.",
    "I focus more on present realities than future possibilities.",
    "I make decisions based on logic rather than feelings.",
    "I value harmony and avoid conflicts.",
    "I keep my environment organized and structured.",
    "I am comfortable adapting to unexpected changes.",
    "I often reflect on abstract theories or concepts.",
    "I get energized when I spend time alone."
]

@app.route("/")
def index():
    # Reset answers when starting
    session["answers"] = []
    return redirect(url_for("question", qid=0))


@app.route("/question/<int:qid>", methods=["GET", "POST"])
def question(qid):
    if request.method == "POST":
        # Save answer
        answer = int(request.form["answer"])
        session["answers"].append(answer)
        session.modified = True

        # Next or results
        if qid + 1 < len(questions):
            return redirect(url_for("question", qid=qid + 1))
        else:
            return redirect(url_for("result"))

    return render_template("index.html", qid=qid, question=questions[qid], total=len(questions))


@app.route("/result")
def result():
    answers = session.get("answers", [])

    # Must have all answers
    if len(answers) != len(questions):
        return redirect(url_for("index"))

    # Convert to model input
    X = scaler.transform([answers])
    prediction = str(model.predict(X)[0]).strip()

     # Full descriptions for all 16 MBTI types
    mbti_descriptions = {
    "INTJ": "The Mastermind: You are visionary, independent, and determined. You love strategy, long-term goals, and working alone to bring your ideas to life.",
    "INTP": "The Thinker: You are curious, analytical, and innovative. You thrive on exploring theories and solving abstract problems.",
    "ENTJ": "The Commander: You are a bold leader — confident and decisive. You enjoy organizing people, creating systems, and driving progress toward big goals.",
    "ENTP": "The Debater: You are energetic, witty, and adventurous. You love challenges, arguments, and brainstorming creative solutions.",
    "INFJ": "The Advocate: You are a deep thinker — compassionate and visionary. You seek meaning in life, value helping others, and often guide people with wisdom.",
    "INFP": "The Mediator: You are gentle, imaginative, and empathetic. You care deeply about harmony, values, and bringing kindness into the world.",
    "ENFJ": "The Protagonist: You are charismatic, inspiring, and supportive. As a natural leader, you motivate others, build strong communities, and value teamwork.",
    "ENFP": "The Campaigner: You are enthusiastic, creative, and adventurous. You love exploring possibilities, making connections, and keeping life exciting.",
    "ISTJ": "The Inspector: You are responsible, organized, and dependable. You value rules, traditions, and work hard to get things done correctly.",
    "ISFJ": "The Nurturer: You are warm, caring, and protective. You are a loyal friend, a great listener, and always look out for others’ well-being.",
    "ESTJ": "The Supervisor: You are efficient, strong-willed, and structured. You like order, clear plans, and taking charge to make sure tasks are done right.",
    "ESFJ": "The Provider: You are sociable, supportive, and generous. You enjoy helping people, maintaining harmony, and making others feel included.",
    "ISTP": "The Virtuoso: You are hands-on, adventurous, and practical. You are a problem-solver who enjoys exploring, fixing, and learning by doing.",
    "ISFP": "The Artist: You are gentle, artistic, and free-spirited. You live in the moment, value beauty, and express yourself creatively.",
    "ESTP": "The Dynamo: You are energetic, bold, and action-oriented. You love excitement, challenges, and learning through real-life experiences.",
    "ESFP": "The Performer: You are outgoing, lively, and fun-loving. You thrive on attention, enjoy socializing, and love making every moment exciting."
}

    description = mbti_descriptions.get(prediction, "No description available.")

   # print("Answers from session:", answers)
    #print("Prediction type:", type(prediction))
    #print("Prediction value:", repr(prediction))


    return render_template("result.html", prediction=prediction, description=description)


if __name__ == "__main__":
    app.run(debug=True)
