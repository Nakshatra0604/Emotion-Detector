import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set resolution (helps face detection)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def get_frame(names=[]):
    """
    Reads a frame from webcam, detects faces and emotions using DeepFace,
    returns the encoded frame and a list of (name, emotion)
    """
    success, frame = cap.read()
    if not success:
        return None, None

    # Convert to RGB for DeepFace
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Analyze emotions
        results = DeepFace.analyze(
            rgb_frame,
            actions=['emotion'],
            enforce_detection=False
        )

        # Handle different output formats
        if isinstance(results, dict) and "instances" in results:
            results = results["instances"]  # multi-face format
        elif not isinstance(results, list):
            results = [results]

    except Exception as e:
        print("DeepFace error:", e)
        results = []

    emotions = []

    for i, result in enumerate(results):
        # Extract face region
        region = result.get('region', {})
        x, y, w, h = (
            region.get('x', 0),
            region.get('y', 0),
            region.get('w', 0),
            region.get('h', 0),
        )

        # Skip invalid detections
        if w == 0 or h == 0:
            continue

        # Get emotion
        dominant_emotion = result.get('dominant_emotion', 'Neutral')

        # Assign name
        name = names[i] if i < len(names) else f"Person {i+1}"
        label = f"{name}: {dominant_emotion}"

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        emotions.append((name, dominant_emotion))

    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes(), emotions
