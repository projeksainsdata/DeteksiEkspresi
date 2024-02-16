from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("last (3).pt")

def prediksi(image, model):
    # Proses deteksi objek menggunakan model dan gambar yang diberikan
    results = model.predict(image)
    
    # Inisialisasi daftar untuk menyimpan hasil deteksi
    detected_objects = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy() # Mendapatkan kotak pembatas objek dalam format numpy array
        
        # Iterasi melalui setiap kotak pembatas
        for box in boxes:
            # Mendapatkan koordinat sudut kotak pembatas sebagai integer
            r = box.xyxy[0].astype(int)
            r = r.tolist()
            # Mendapatkan kelas objek dan mengonversi ke label yang sesuai
            cls = result.names[int(box.cls[0])]
            if str(cls) == "玫瑰" or str(cls) == "玫瑰 0.88":
                cls = "mawar"
            elif str(cls) == "向日葵 0.92":
                cls = "sunflower"
            
            # Membuat objek deteksi dengan properti yang sesuai
            detected_object = {
                "x": r[0],
                "y": r[1],
                "width": r[2] - r[0],
                "height": r[3] - r[1],
                "label": str(cls)
            }
            
            # Menambahkan objek deteksi ke dalam daftar hasil deteksi
            detected_objects.append(detected_object)
    
    # Mengembalikan daftar hasil deteksi
    return detected_objects



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Ambil data gambar dari request JSON
    data = request.get_json()
    image_data = data['image']

    # Decode data gambar dari base64
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Proses gambar menggunakan model prediksi
    detection_results = prediksi(image, model)
    
    return jsonify(detection_results)


if __name__ == '__main__':
    app.run(debug=True)


