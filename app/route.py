from flask import Flask, render_template, Response, request
import cv2
from deepface import DeepFace
import pandas as pd
import datetime

app = Flask(__name__)

people = {}  # store names and emotions
cap = None   # webcam

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Step 1: Number of people
@app.route('/start', methods=['POST'])
def start():
    num_people = int(request.form['num_people'])
    return render_template('names.html', num_people=num_people)

# Step 2: Collect names
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global people, cap
    people = {}
    i = 0
    while f"person{i}" in request.form:
        people[i] = {"name": request.form[f"person{i}"], "emotions": []}
        i += 1
    cap = cv2.VideoCapture(0)
    return render_template('video.html')

# Stream video + detect emotions
def generate_frames():
    global people, cap
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # DeepFace sometimes returns a dict, sometimes a list
                if not isinstance(results, list):
                    results = [results]

                for idx, face in enumerate(results):
                    region = face.get('region', {})
                    x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)

                    if w == 0 or h == 0:
                        continue

                    dominant_emotion = face.get('dominant_emotion', 'neutral')

                    if idx in people:
                        name = people[idx]['name']
                        people[idx]['emotions'].append(dominant_emotion)

                        # ✅ Save log entry
                        with open("emotions_log.csv", "a") as f:
                            f.write(f"{datetime.datetime.now()},{name},{dominant_emotion}\n")

                        # Draw on frame
                        cv2.putText(frame, f"{name}: {dominant_emotion}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            except Exception as e:
                print("Error:", e)

            # Encode frame to send
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Dashboard - show CSV data
@app.route('/dashboard')
def dashboard():
    try:
        df = pd.read_csv("emotions_log.csv", names=["time", "name", "emotion"])
        return render_template("dashboard.html", data=df.to_dict(orient="records"))
    except:
        return "No data logged yet!"

if __name__ == "__main__":
    app.run(debug=True)
