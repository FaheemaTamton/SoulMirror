# SoulMirror – MBTI Personality Prediction System

> “See Yourself Reflected Back!”

SoulMirror is a Machine Learning powered web application that predicts a user's MBTI personality type using a short personality questionnaire.

The system uses a trained Random Forest Classification model built with Scikit-learn and deployed through a Flask web application.

---

# Features

• Predicts all 16 MBTI personality types  
• Machine Learning based prediction system  
• Interactive Flask web application  
• Modern responsive UI with custom CSS  
• Uses personality dataset for model training  
• Fast real-time personality prediction  
• Displays detailed personality descriptions  

---

# Tech Stack

## Frontend
• HTML5  
• CSS3  

## Backend
• Python  
• Flask  

## Machine Learning
• Scikit-learn  
• Random Forest Classifier  
• StandardScaler  
• Pandas  
• NumPy  
• Joblib  

---

# Project Structure

```bash
SoulMirror/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── 16P.csv
│   └── 16p-Mapping.txt
│
├── models/
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html



# How It Works

Step 1 — User Answers Questions
The user answers 10 personality-based questions through the web interface.

Step 2 — Data Preprocessing
The answers are scaled using StandardScaler.

Step 3 — ML Prediction
The trained Random Forest model predicts the user's MBTI personality type.

Step 4 — Result Display
The predicted personality type and description are displayed on the result page.

# MBTI Personality Types Supported

• INTJ – The Mastermind
• INTP – The Thinker
• ENTJ – The Commander
• ENTP – The Debater
• INFJ – The Advocate
• INFP – The Mediator
• ENFJ – The Protagonist
• ENFP – The Campaigner
• ISTJ – The Inspector
• ISFJ – The Nurturer
• ESTJ – The Supervisor
• ESFJ – The Provider
• ISTP – The Virtuoso
• ISFP – The Artist
• ESTP – The Dynamo
• ESFP – The Performer



# Generate Model Files

Run the following command to generate the trained model files locally:

python train.py
User Interface

# The application includes:
• Animated UI
• Dark modern theme
• Interactive radio buttons
• Responsive card layout
• Result visualization pages



# Author

FAHEEMA TAMTON
AI & ML Engineer
🔗 [Portfolio](https://your-portfolio-link.com)  



# License
This project is licensed under the MIT License.