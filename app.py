from flask import Flask, render_template, request, session, url_for, redirect
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import mxnet as mx
import numpy as np
import uuid
from PIL import Image
from flask import send_from_directory



app = Flask(__name__, static_folder="C:/python/static")
@app.route('/pool/<path:filename>')
def custom_static(filename):
    return send_from_directory(r'C:\python\pool', filename)

@app.route('/static_images/<path:filename>')
def processed_static(filename):
    return send_from_directory(r'C:\python\static', filename)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = r'C:\python\pool'
STATIC_FOLDER = r'C:\python\static'
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
def calculate_similarity(emb_a, emb_b):
    similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    return similarity
def alignforcheck(selected_face,filename,images):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(path)
    faces = detector.get(img)
    for face in faces:
             landmarks = face['kps'].astype(np.int) 
             aligned_filename = f"aligned_{selected_face}_{filename}"   # Name contains the selected face index
             aligned_path =os.path.join(STATIC_FOLDER, aligned_filename)
             aligned_img = norm_crop(img, landmarks,112,'arcface')                            
             cv2.imwrite(aligned_path, aligned_img)
             images.append(aligned_filename)
             uploaded_images = images
             return uploaded_images

def extract_embedding(embedder, face_data):
    try:        
        if face_data and 'embedding' in face_data:
            embedding = face_data['embedding']
            return embedding
        else:
            print("No faces detected.")  # Debug log
            return None
    except Exception as e:
        print("Error during embedding extraction:", e)  # Debug log
        return None

def load_model_for_embedding():
    try:
        model_name = 'arcface_r100_v1' # Use the face recognition model
        embedder = FaceAnalysis(model=model_name)
        embedder.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        return embedder
       
    except Exception as e:
        print("Error during embedder model initialization:", e)
        return None
detector = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    showAlert=False
    images = []
    message = ""
    sim_message=""
    faces_length = 0
    images_length=0
    selected_option = request.form.get('face_selection')

    if not detector:
        return "Error: Model is not initialized. Check server logs."

    uploaded_images = session.get('uploaded_images', [])
    check_uploaded_images = session.get('check_uploaded_images', [])
   
    selected_face = int(request.form.get('selected_face', -2))
    


    if request.method == 'POST':
        action = request.form.get('action')
        

        if action == 'Upload':
            
            for image_name in ['image1', 'image2']:
                file = request.files.get(image_name)
                check_uploaded_images.append(file.filename)
                

                if file and file.filename:
                    if allowed_file(file.filename):
                        filename = file.filename
                        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)    
                        try:
                            file.save(path)
                        except Exception as e:
                            message += f"Failed to save {file.filename} due to error: {str(e)}"                    
                        uploaded_images.append(filename)                        
                    else:
                        message += f"Invalid file format for {filename}. "
            session['check_uploaded_images'] = check_uploaded_images

                       
            
            # Detect faces immediately after uploading to fill the combo box
            for filename in uploaded_images:
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                
                img = cv2.imread(path)                
                faces = detector.get(img)
                if faces:
                    faces_length += len(faces)
                
        elif action in ["Detect", "Align"]:
            for filename in uploaded_images:
                if "aligned" in filename or "detected" in filename :
                  path=  os.path.join(STATIC_FOLDER, filename)
                else:    
                 path = os.path.join(app.config['UPLOAD_FOLDER'], filename)                
                img = cv2.imread(path)                
                faces = detector.get(img)
                message += f"{len(faces)} detected faces. "
        
                # if faces and 0 <= selected_face < len(faces):
                if faces:
                    
                    # face = faces[selected_face]
                    if action == "Align":
                        if selected_face == -2:
                              face_count = 0
                              for face in faces:                         
                               landmarks = face['kps'].astype(np.int)                   
                               aligned_filename = f"aligned_{face_count}_{filename}"
                               aligned_path = os.path.join(STATIC_FOLDER, aligned_filename)
                               aligned_img = norm_crop(img, landmarks,112,'arcface')                                                     
                               cv2.imwrite(aligned_path, aligned_img)                              
                               images.append(aligned_filename)
                               face_count += 1


                        elif 0 <= selected_face < len(faces):
                            face = faces[selected_face]
                            landmarks = face['kps'].astype(np.int)
                            aligned_filename = f"aligned_{selected_face}_{filename}"   # Name contains the selected face index
                            aligned_path = os.path.join(STATIC_FOLDER, aligned_filename)
                            aligned_img = norm_crop(img, landmarks,112,'arcface')                            
                            cv2.imwrite(aligned_path, aligned_img)
                            images.append(aligned_filename)
                       
                    
                    elif action == "Detect":
                        for face in faces:
                            landmarks = face['kps'].astype(np.int)  
                            for point in landmarks:
                                cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -2)
                        
                        detected_filename = "detected_" + filename
                        detected_path = os.path.join(STATIC_FOLDER, detected_filename)
                        # message += f"path {detected_path}. "
                        cv2.imwrite(detected_path, img)
                        images.append(detected_filename)
                else:
                    images.append(filename)

            uploaded_images = images
            
        elif action == "Clear":
            uploaded_images = []
            check_uploaded_images = []
        
        elif action=="Compare":
           showAlert=True
           check_uploaded_images = [x for x in check_uploaded_images if x != '']
           uploaded_images=alignforcheck(selected_face,check_uploaded_images[0],images)
           uploaded_images=alignforcheck(selected_face,check_uploaded_images[1],images)
           if len(check_uploaded_images) ==2:
               embedder = load_model_for_embedding()
               if embedder:
                   image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in check_uploaded_images]
                   embeddings = []
                   if len(uploaded_images)==2:
                    for path in image_paths:                        
                       img = cv2.imread(path)                
                       faces = embedder.get(img)   
                       if faces:
                           embedding = extract_embedding(embedder, faces[0])
                           if embedding is not None:  # Assuming there's only one face in the image
                            embeddings.append(embedding)
                           # uploaded_images=alignforcheck(selected_face,path,images)      
                           else:
                               print("No embedding extracted.")  # Debug log
                       else:
                           print("No faces detected.")  # Debug log
                           message = "No faces detected in one or both images."
                           break
                                    
                   if len(embeddings) == 2:
                       similarity = calculate_similarity(embeddings[0], embeddings[1])
                       message = f"Similarity: {similarity:.4f}"
                       if similarity>=0.6518:
                           sim_message="THIS IS PROBABLY THE SAME PERSON"
                       else:
                            sim_message="THIS IS PROBABLY NOT THE SAME PERSON"

                   elif uploaded_images!=2:
                        message = "choose 2 images!"
                   else:
                       message = "Error: Failed to extract embeddings from images."
               else:
                    message = "Error: Embedder model not initialized."
           else:
                message = "Select exactly 2 images for comparison."   
        elif action=="Check":
            
            check_uploaded_images = [x for x in check_uploaded_images if x != '']
            uploaded_images=alignforcheck(selected_face,check_uploaded_images[0],images)
            if len(check_uploaded_images) ==1:
             embedder = load_model_for_embedding()
             user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], check_uploaded_images[0])
             user_img = cv2.imread(user_image_path)
             user_faces = embedder.get(user_img)
             user_embedding = extract_embedding(embedder, user_faces[0])
             max_similarity = -1
             most_similar_image = None
             with os.scandir(UPLOAD_FOLDER) as entries:
              for entry in entries:
               if  entry.name != check_uploaded_images[0] and check_uploaded_images[0] not in entry.name:
                 if check_uploaded_images[0] not in entry.name:


                  if entry.is_file()  and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                   with Image.open(entry.path) as img:
                    if embedder:
                     img = cv2.imread(entry.path)
                     faces = embedder.get(img)
                     if faces:
                        embedding = extract_embedding(embedder, faces[0])
                        if embedding is not None:
                              similarity = calculate_similarity(user_embedding, embedding)
                              if similarity > max_similarity:
                                  max_similarity = similarity
                                  most_similar_image = entry.name
            if most_similar_image:
                # uploaded_images.append(most_similar_image)
                 uploaded_images=alignforcheck(selected_face,most_similar_image,images)
                 message = f"The most similar image is {most_similar_image} with similarity of {max_similarity:.4f}"
                 
                              
                                  


                      

                    
                    
                
                     
        images_length = len(uploaded_images)
        
        session['uploaded_images'] = uploaded_images
        session['check_uploaded_images'] = check_uploaded_images

    

         
          

    images = uploaded_images
    

    return render_template('image.html', images=images, message=message, faces_length=faces_length, images_length=images_length,showAlert=showAlert,sim_message=sim_message)



if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
