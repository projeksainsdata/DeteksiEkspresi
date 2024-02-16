from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("last (3).pt")
camera = cv2.VideoCapture(0)

def prediksi(image, model):
    results = model.predict(image)
    for result in results:
        boxes = result.boxes.cpu().numpy() # get boxes on cpu in numpy
        for box in boxes: # iterate boxes
            r = box.xyxy[0].astype(int) # get corner points as int
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0,255,0), 2) # draw boxes on image
            cls = result.names[int(box.cls[0])]
            cv2.putText(image, str(cls), (r[0] + 5, r[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
    return image

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame = prediksi(frame, model)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
