from flask import Flask, render_template, request, session, url_for, redirect
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import numpy as np
import uuid
from PIL import Image
from flask import send_from_directory

APP_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)


@app.route("/pool/<path:filename>")
def custom_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static_images/<path:filename>")
def processed_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


app.secret_key = "your_secret_key"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    try:
        face_analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
        face_analyzer.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        return face_analyzer

    except Exception as e:
        print("Error during model initialization:", e)
        return None


# Load model on startup
def calculate_similarity(emb_a, emb_b):
    similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    return similarity


def alignforcheck(selected_face, filename, uploaded_images):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = cv2.imread(path)
    faces = detector.get(img)
    for face in faces:
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"  # Name contains the selected face index
        aligned_path = os.path.join(STATIC_FOLDER, aligned_filename)
        if not os.path.exists(aligned_path):
            aligned_img = norm_crop(img, landmarks, 112, "arcface")
            cv2.imwrite(aligned_path, aligned_img)
        uploaded_images.append(aligned_filename)
        return uploaded_images


def extract_embedding(embedder, face_data):
    try:
        if face_data and "embedding" in face_data:
            embedding = face_data["embedding"]
            return embedding
        else:
            print("No faces detected.")  # Debug log
            return None
    except Exception as e:
        print("Error during embedding extraction:", e)  # Debug log
        return None


def load_model_for_embedding():
    try:
        model_name = "arcface_r100_v1"  # Use the face recognition model
        embedder = FaceAnalysis(model=model_name)
        embedder.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        return embedder

    except Exception as e:
        print("Error during embedder model initialization:", e)
        return None


detector = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    showAlert = False
    images = []
    
    unaligned_images = []
    message = ""
    messages=[]
    errors=[]
    sim_message = ""
    images_length = 0
    selected_option = request.form.get("face_selection")

    if not detector:
        return "Error: Model is not initialized. Check server logs."

    uploaded_images = session.get("uploaded_images", [])
    check_uploaded_images = session.get("check_uploaded_images", [])
    faces_length = session.get("faces_length", [0, 0])
    current_images = session.get("current_images", [])

    selected_face = int(request.form.get("selected_face", -2))
    combochange = session.get("combochange", selected_face)
    if selected_face != -2:
        combochange = selected_face
        session["combochange"] = combochange

    check_uploaded_images = session.get("check_uploaded_images", [])

    selected_face = int(request.form.get("selected_face", -2))

    if request.method == "POST":
        action = request.form.get("action")
        if action == "Upload":
            current_images = []
            for image_name in ["image1", "image2"]:
                file = request.files.get(image_name)
                check_uploaded_images.append(file.filename)

                if file and file.filename:
                    if allowed_file(file.filename):
                        session.pop("uploaded_images", None)
                        filename = file.filename
                        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                        try:
                            file.save(path)
                        except Exception as e:
                            errors.append(
                                f"Failed to save {file.filename} due to error: {str(e)}"
                            )
                        uploaded_images.append(filename)
                        current_images.append(filename)
                    else:
                        errors.append(f"Invalid file format for {filename}. ")
            session["check_uploaded_images"] = check_uploaded_images

            # Detect faces immediately after uploading to fill the combo box
            i = 0
            for filename in current_images:
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                img = cv2.imread(path)
                faces = detector.get(img)
                if faces:
                    faces_length[i] = len(faces)
                i = i + 1
        elif action in ["Detect", "Align"]:
            current_images = session.get("current_images")
            for filename in current_images:
                if "aligned" in filename or "detected" in filename:
                    path = os.path.join(STATIC_FOLDER, filename)
                else:
                    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                img = cv2.imread(path)
                faces = detector.get(img)
                messages.append(f"{len(faces)} detected faces. ")

                # if faces and 0 <= selected_face < len(faces):
                if faces:
                    # face = faces[selected_face]
                    if action == "Align":
                        if selected_face == -2:
                            face_count = 0
                            for face in faces:
                                landmarks = face["kps"].astype(int)
                                aligned_filename = f"aligned_{face_count}_{filename}"
                                aligned_path = os.path.join(
                                    STATIC_FOLDER, aligned_filename
                                )
                                aligned_img = norm_crop(img, landmarks, 112, "arcface")
                                cv2.imwrite(aligned_path, aligned_img)
                                uploaded_images.append(aligned_filename)
                                face_count += 1

                        elif 0 <= selected_face < len(faces):
                            face = faces[selected_face]
                            landmarks = face["kps"].astype(int)
                            aligned_filename = f"aligned_{selected_face}_{filename}"  # Name contains the selected face index
                            aligned_path = os.path.join(STATIC_FOLDER, aligned_filename)
                            aligned_img = norm_crop(img, landmarks, 112, "arcface")
                            cv2.imwrite(aligned_path, aligned_img)
                            images.append(aligned_filename)

                            # unaligned_filename = f"unaligned_{selected_face}_{filename}" # Name contains the selected face index
                            # unaligned_path = os.path.join(app.config['UPLOAD_FOLDER'], unaligned_filename)
                            # bbox = face['bbox'].astype(np.int)
                            # x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                            # unaligned_face = img[y:y2, x:x2]
                            # cv2.imwrite(unaligned_path, unaligned_face)
                            # unaligned_images.append(unaligned_filename)

                    elif action == "Detect":
                        for face in faces:
                            landmarks = face["kps"].astype(int)
                            for point in landmarks:
                                cv2.circle(
                                    img,
                                    (int(point[0]), int(point[1])),
                                    2,
                                    (0, 255, 0),
                                    -2,
                                )

                        detected_filename = "detected_" + filename
                        detected_path = os.path.join(STATIC_FOLDER, detected_filename)
                        # message += f"path {detected_path}. "
                        cv2.imwrite(detected_path, img)
                        uploaded_images.append(detected_filename)
                else:
                    uploaded_images.append(filename)

        elif action == "Clear":
            uploaded_images = []
            check_uploaded_images = []
            combochange = selected_face
            session["combochange"] = combochange
            current_images = []
            faces_length = [0, 0]

        # START 1

        elif action == "Compare":
            image1 = current_images[0]
            image2 = current_images[1]
            check_uploaded_images = [
                image1,
                image2,
            ]
            face_num_1 = int(request.form.get("face_num1"))
            face_num_2 = int(request.form.get("face_num2"))
            combochanges = [face_num_1, face_num_2]
            check_uploaded_images = [x for x in check_uploaded_images if x != ""]
            embeddings = []
            embedder = load_model_for_embedding()
            image_paths = [
                os.path.join(app.config["UPLOAD_FOLDER"], filename)
                for filename in check_uploaded_images
            ]
            for i in range(2):
                uploaded_images = alignforcheck(
                    combochanges[i], check_uploaded_images[i], images
                )
                if len(check_uploaded_images) == 2:
                    if embedder:
                        img = cv2.imread(image_paths[i])
                        faces = embedder.get(img)
                        if faces:
                            if combochanges[i] == -2:
                                embedding = extract_embedding(
                                    embedder, faces[0]
                                )
                            elif len(faces) == 1:
                                embedding = extract_embedding(
                                    embedder, faces[0]
                                )
                            else:
                                embedding = extract_embedding(
                                    embedder, faces[combochanges[i]]
                                )
                            if (
                                embedding is not None
                            ):  # Assuming there's only one face in the image
                                embeddings.append(embedding)
                            # uploaded_images=alignforcheck(selected_face,path,images)
                            else:
                                print("No embedding extracted.")  # Debug log
                        else:
                            print("No faces detected.")  # Debug log
                            errors.append("No faces detected in one or both images.");
                            # break

                        
                    else:
                        errors.append("Error: Embedder model not initialized.")
                else:
                    errors.append("Select exactly 2 images for comparison.")

            if len(embeddings) == 2:
                similarity = calculate_similarity(
                    embeddings[0], embeddings[1]
                )
                messages.append(f"Similarity: {similarity:.4f}")
                if similarity >= 0.6518:
                    messages.append("THIS IS PROBABLY THE SAME PERSON")
                else:
                    messages.append("THIS IS PROBABLY NOT THE SAME PERSON")

            elif len(check_uploaded_images) != 2:
                errors.append("choose 2 images!")
            else:
                errors.append("Error: Failed to extract embeddings from images.")


        elif action == "Check":
            check_uploaded_images = [x for x in check_uploaded_images if x != ""]
            if combochange == -2:
                uploaded_images = alignforcheck(
                    selected_face, check_uploaded_images[0], images
                )
            else:
                uploaded_images = alignforcheck(
                    combochange, check_uploaded_images[0], images
                )
            if len(check_uploaded_images) == 1:
                embedder = load_model_for_embedding()
                #  if(combochange==-2):
                user_image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], check_uploaded_images[0]
                )
                # else:
                # user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unaligned_filename[0])
                user_img = cv2.imread(user_image_path)
                user_faces = embedder.get(user_img)
                if combochange == -2:
                    user_embedding = extract_embedding(embedder, user_faces[0])
                else:
                    user_embedding = extract_embedding(
                        embedder, user_faces[combochange]
                    )

                max_similarity = -1
                most_similar_image = None
                with os.scandir(UPLOAD_FOLDER) as entries:
                    for entry in entries:
                        if (
                            entry.name != check_uploaded_images[0]
                            and check_uploaded_images[0] not in entry.name
                        ):
                            if check_uploaded_images[0] not in entry.name:
                                if entry.is_file() and entry.name.lower().endswith(
                                    (
                                        ".png",
                                        ".jpg",
                                        ".jpeg",
                                        ".gif",
                                        ".bmp",
                                        ".tif",
                                        ".tiff",
                                    )
                                ):
                                    with Image.open(entry.path) as img:
                                        if embedder:
                                            img = cv2.imread(entry.path)
                                            faces = embedder.get(img)
                                            if faces:
                                                embedding = extract_embedding(
                                                    embedder, faces[0]
                                                )
                                                if embedding is not None:
                                                    similarity = calculate_similarity(
                                                        user_embedding, embedding
                                                    )
                                                    if similarity > max_similarity:
                                                        max_similarity = similarity
                                                        most_similar_image = entry.name
            if most_similar_image:
                # uploaded_images.append(most_similar_image)
                # if(combochange==-2):
                uploaded_images = alignforcheck(
                    selected_face, most_similar_image, images
                )
                # else:
                #    uploaded_images=alignforcheck(combochange,most_similar_image,images)
                message = f"The most similar image is {most_similar_image} with similarity of {max_similarity:.4f}"
        
        
        images_length = len(uploaded_images)
        session["current_images"] = current_images
        session["uploaded_images"] = uploaded_images
        session["check_uploaded_images"] = check_uploaded_images
        session["faces_length"] = faces_length

    images = uploaded_images

    return render_template(
        "image.html",
        images=images,
        current=current_images,
        faces_length=faces_length,
        errors=errors,
        messages=messages,
    )


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
