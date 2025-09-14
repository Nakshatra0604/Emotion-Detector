Emotion Detector (Face Sentiment Analysis)

A real-time Emotion Detection System using Python, OpenCV, and Deep Learning to analyze facial expressions from live video feed.

Overview

--This project detects human emotions (e.g., happy, sad, angry, neutral) by analyzing facial expressions using a pre-trained deep learning model.

Tech Stack

- Python
- OpenCV
- TensorFlow
- Flask (for web interface)

Features

- Real-time emotion detection from webcam
- Emotion logging (emotions_log.csv)
- Interactive dashboard built with Flask

Project Structure

face_sentiment/
│── app/
│ ├── route.py                        # Flask routes
│ ├── templates/                      # HTML files (dashboard, index, video)
│── face_sentiment.py                 # main script
│── analyze_emotions.py               # emotion analysis logic
│── emotions_log.csv                  # logs
│── requirements.txt                  # dependencies

Usage
Run:

--python run.py

Access the web interface at:

--http://127.0.0.1:5000/

Future Improvements:

--Support multi-face detection

--Train custom models for higher accuracy

--Deploy as a cloud-based API
