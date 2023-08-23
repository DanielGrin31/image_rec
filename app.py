from flask import Flask, render_template, request, session, url_for, redirect
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import mxnet as mx
import numpy as np

app = Flask(__name__, static_folder="C:/python/static")
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    try:
        ctx = mx.cpu()
        face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
        face_analyzer.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        return face_analyzer
        
    except Exception as e:
        print("Error during model initialization:", e)
        return None

# Load model on startup
detector = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    images = []
    message = ""

    if not detector:
        return "Error: Model is not initialized. Check server logs."

    uploaded_images = session.get('uploaded_images', [])

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'Upload':
            for image_name in ['image1', 'image2']:
                file = request.files.get(image_name)

                if file and file.filename:
                    if allowed_file(file.filename):
                        filename = file.filename
                        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(path)
                        uploaded_images.append(filename)
                    else:
                        message += f"Invalid file format for {filename}. "

        elif action in ["Detect", "Align"]:
            for filename in uploaded_images:
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                
                img = cv2.imread(path)                
                faces = detector.get(img)
                message += f"{len(faces)} detected faces. "
        
                if faces:
                    if action == "Align":
                        face_count = 0
                        for face in faces:                         
                            landmarks = face['kps'].astype(np.int)                   
                            aligned_filename = f"aligned_{face_count}_{filename}"
                            aligned_path = os.path.join(app.config['UPLOAD_FOLDER'], aligned_filename)
                            aligned_img = norm_crop(img, landmarks)
                            cv2.imwrite(aligned_path, aligned_img)
                            images.append(aligned_filename)
                            face_count += 1
                    
                    elif action == "Detect":
                        for face in faces:
                            landmarks = face['kps'].astype(np.int)  
                            for point in landmarks:
                                cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                        
                        detected_filename = "detected_" + filename
                        detected_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
                        cv2.imwrite(detected_path, img)
                        images.append(detected_filename)
                else:
                    images.append(filename)

            uploaded_images = images

        elif action == "Clear":
            uploaded_images = []

        session['uploaded_images'] = uploaded_images

    images = uploaded_images
    return render_template('image.html', images=images, message=message)

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5050)
    except Exception as e:
        print(f"Error: {e}")
