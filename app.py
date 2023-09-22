from flask import Flask, render_template, request, session, url_for, redirect
import os
import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import numpy as np
from PIL import Image
from image_helper import ImageHelper
from model_loader import ModelLoader
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


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR

detector = ModelLoader.load_detector()
embedder = ModelLoader.load_embedder()
helper = ImageHelper(detector,embedder, UPLOAD_FOLDER, STATIC_FOLDER)


@app.route("/", methods=["GET", "POST"])
def index():
    images = []
    messages = []
    errors = []

    if not detector:
        return "Error: Model is not initialized. Check server logs."

    uploaded_images = session.get("uploaded_images", [])
    faces_length = session.get("faces_length", [0, 0])
    current_images = session.get("current_images", [])

    if request.method == "POST":
        face_num_1 = int(request.form.get("face_num1"))
        face_num_2 = int(request.form.get("face_num2"))
        combochanges = [face_num_1, face_num_2]
        action = request.form.get("action")
        if action == "Upload":
            current_images = []
            for image_name in ["image1", "image2"]:
                file = request.files.get(image_name)

                if file and file.filename:
                    if ImageHelper.allowed_file(file.filename):
                        session.pop("uploaded_images", None)
                        path = os.path.join(UPLOAD_FOLDER, file.filename)
                        try:
                            file.save(path)
                        except Exception as e:
                            errors.append(
                                f"Failed to save {file.filename} due to error: {str(e)}"
                            )
                        uploaded_images.append(file.filename)
                        current_images.append(file.filename)
                    else:
                        errors.append(f"Invalid file format for {file.filename}. ")

            # Detect faces immediately after uploading to fill the combo box

            for i in range(len(current_images)):
                faces_length[i] = helper.create_aligned_images(
                    current_images[i], uploaded_images
                )

        elif action in ["Detect", "Align"]:
            for filename in current_images:
                if "aligned" in filename or "detected" in filename:
                    path = os.path.join(STATIC_FOLDER, filename)
                else:
                    path = os.path.join(UPLOAD_FOLDER, filename)

                    if action == "Align":
                        face_count = helper.create_aligned_images(
                            filename, uploaded_images
                        )
                        messages.append(f"{face_count} detected faces in {filename}.")

                    elif action == "Detect":
                        face_count = helper.detect_faces_in_image(
                            filename, uploaded_images
                        )
                        messages.append(f"{face_count} detected faces in {filename}. ")

        elif action == "Clear":
            uploaded_images = []
            current_images = []
            faces_length = [0, 0]

        elif action == "Compare":
            embeddings = []
            image_paths = [
                os.path.join(UPLOAD_FOLDER, filename) for filename in current_images
            ]
            for i in range(len(current_images)):
                if len(current_images) == 2:
                    # Generate an embedding for a specific face(first by default) in each image
                    embedding,temp_err=helper.generate_embeddings(image_paths[i],combochanges[i])
                    # Add the errors and embeddings from the helper function to the local variables
                    errors=errors+temp_err;
                    if embedding is not None:
                        embeddings.append(embedding);
                    else:
                        print("No embedding extracted.")  # Debug log
                else:
                    errors.append("Select exactly 2 images for comparison.")

            if len(embeddings) == 2:
                # Calculate similarity between the two images
                similarity = ImageHelper.calculate_similarity(
                    embeddings[0], embeddings[1]
                )
                messages.append(f"Similarity: {similarity:.4f}")
                if similarity >= 0.6518:
                    messages.append("THIS IS PROBABLY THE SAME PERSON")
                else:
                    messages.append("THIS IS PROBABLY NOT THE SAME PERSON")

            elif len(current_images) != 2:
                errors.append("choose 2 images!")
            else:
                errors.append("Error: Failed to extract embeddings from images.")

        elif action == "Check":
            most_similar_image = None
            if len(current_images) == 1:
                embedder = ModelLoader.load_embedder()
                user_image_path = os.path.join(UPLOAD_FOLDER, current_images[0])
                # else:
<<<<<<< HEAD
                user_embedding=helper.generate_embedding(user_image_path,combochanges[0]);
=======
                # user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unaligned_filename[0])
                user_img = cv2.imread(user_image_path)
                user_faces = embedder.get(user_img)
                if combochanges[0] == -2:
                    user_embedding = extract_embedding(embedder, user_faces[0])
                else:
                    user_embedding = extract_embedding(
                        embedder, user_faces[combochange]
                    )
>>>>>>> ab2820c (Update README)

                max_similarity = -1
                most_similar_image = None
                most_similar_path = None
                with os.scandir(UPLOAD_FOLDER) as entries:
                    for entry in entries:
                        if (
                            entry.name != current_images[0]
                            and current_images[0] not in entry.name
                        ):
                            if current_images[0] not in entry.name:
                                if entry.is_file() and entry.name.lower().endswith(
                                    (ImageHelper.ALLOWED_EXTENSIONS)
                                ):
                                    with Image.open(entry.path) as img:
                                        if embedder:
                                            img = cv2.imread(entry.path)
                                            faces = embedder.get(img)
                                            if faces:
<<<<<<< HEAD
                                                embedding = (ImageHelper.extract_embedding(faces[0]))
                                                if embedding is not None:
                                                    similarity = ImageHelper.calculate_similarity(
                                                        user_embedding, embedding
                                                    )
                                                    if similarity > max_similarity:
                                                        max_similarity = similarity
                                                        most_similar_image = entry.name
                                                        most_similar_path = entry.path
            if most_similar_image:
                messages.append(
                    f"The most similar image is {most_similar_image} with similarity of {max_similarity:.4f}")

=======
                                                for face in faces:
                                                    embedding = extract_embedding(
                                                        embedder, face
                                                    )
                                                    if embedding is not None:
                                                        similarity = calculate_similarity(
                                                            user_embedding, embedding
                                                        )
                                                        if similarity > max_similarity:
                                                            max_similarity = similarity
                                                            most_similar_image = entry.name
            if most_similar_image!=None and most_similar_image:
                # uploaded_images.append(most_similar_image)
                # if(combochange==-2):
                uploaded_images = alignforcheck(
                    selected_face, most_similar_image, images
                )
                # else:
                #    uploaded_images=alignforcheck(combochange,most_similar_image,images)
                messages.append(f"The most similar image is {most_similar_image} with similarity of {max_similarity:.4f}");
        
        
        images_length = len(uploaded_images)
>>>>>>> ab2820c (Update README)
        session["current_images"] = current_images
        session["uploaded_images"] = uploaded_images
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
