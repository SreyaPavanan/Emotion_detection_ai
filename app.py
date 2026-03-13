from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

current_emotion = "Neutral"

model = load_model("emotion_model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = None


def generate_frames():
    global camera
    global current_emotion

    while camera is not None:

        success, frame = camera.read()      

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(48,48))
            face = face/255.0
            face = np.reshape(face,(1,48,48,1))

            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]
            current_emotion=emotion

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,emotion,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/emotion')
def get_emotion():
    return current_emotion


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/start')
def start_camera():
    global camera
    camera = cv2.VideoCapture(0)
    return "Camera Started"


@app.route('/stop')
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return "Camera Stopped"


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)