from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import os
import requests as req
from image_helper import ImageHelper
from image_embedding_manager import ImageEmbeddingManager
from model_loader import ModelLoader
from flask_cors import CORS, cross_origin
from flask import send_from_directory
APP_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")
# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/pool/<path:filename>")
def custom_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static_images/<path:filename>")
def processed_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route("/api/upload", methods=["POST"])
def upload_image():
    images = []
    errors = []
    faces_length = [0, 0]
    current_images = []
    for image_name in ["image1", "image2"]:
        file = request.files.get(image_name)

        if file and file.filename:
            if ImageHelper.allowed_file(file.filename):
                session.pop("uploaded_images", None)
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                try:
                    file.save(path)
                    # Generate the embeddings for all faces and store them for future indexing
                    temp_err=helper.generate_all_emb(path,file.filename);
                    errors=errors+temp_err;

                    if(len(errors)>0):
                        os.remove(path);
                    else:
                        current_images.append(file.filename)
                        images.append(file.filename)
                except Exception as e:
                    errors.append(
                        f"Failed to save {file.filename} due to error: {str(e)}"
                    )
                
            else:
                errors.append(f"Invalid file format for {file.filename}. ")

    if(len(errors)==0):
    # Detect faces immediately after uploading to fill the combo box
        for i in range(len(current_images)):
            faces_length[i] = helper.create_aligned_images(current_images[i], images)
    return jsonify({"images": images, "faces_length": faces_length, "errors": errors})


@app.route("/api/align", methods=["POST"])
def align_image():
    uploaded_images = request.form.getlist("images")
    faces_length = [0, 0]
    messages = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            face_count = helper.create_aligned_images(filename, images)
            faces_length[i]=face_count;
            messages.append(f"{face_count} detected faces in {filename}.")
        else:
            errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect_image():
    uploaded_images = request.form.getlist("images")
    faces_length = [0, 0]
    messages = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            face_count = helper.detect_faces_in_image(
                filename, images
                )
            faces_length[i]=face_count;
            messages.append(f"{face_count} detected faces in {filename}.")
        else:
            errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/compare", methods=["POST"])
def compare_image():
    uploaded_images = request.form.getlist("images");
    combochanges=[int(x) for x in request.form.getlist("selected_faces")];
    embeddings = []
    messages = []
    errors = []
    image_paths = [
        os.path.join(UPLOAD_FOLDER, filename) for filename in uploaded_images
    ]
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename=f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding=manager.get_embedding_by_name(filename)
            if len(embedding)>0:
                embeddings.append(embedding);
            else:
                # Generate an embedding for a specific face(first by default) in each image
                embedding, temp_err = helper.generate_embedding(
                    image_paths[i], combochanges[i]
                )
                # Add the errors and embeddings from the helper function to the local variables
                errors = errors + temp_err
                if embedding is not None:
                    embeddings.append(embedding)
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

    elif len(uploaded_images) != 2:
        errors.append("choose 2 images!")
    else:
        errors.append("Error: Failed to extract embeddings from images.")
    
    return jsonify(
            {
                "errors": errors,
                "messages": messages,
            });

@app.route("/api/check", methods=["POST"])
def find_similar_image():
    most_similar_image = None
    messages=[];
    errors=[];
    similarity = -1
    face_length=0;
    current_image = request.form.get("image");
    selected_face=int(request.form.get("selected_face"));
    if len(current_image) > 0  :
        (
            most_similar_image,
            most_similar_face_num,
            similarity,
            temp_err,
        ) = helper.get_most_similar_image(selected_face, current_image)
        errors = errors + temp_err
    else:
        errors.append("no images selected for check")
    if most_similar_image:
        messages.append(
            f"The most similar face is no. {most_similar_face_num+1} in image {most_similar_image} with similarity of {similarity:.4f}"
        )
        face_length = helper.create_aligned_images(most_similar_image, [])
    return jsonify(
    {
        "image":most_similar_image,
        "face":most_similar_face_num,
        "face_length":face_length,
        "errors": errors,
        "messages": messages,
    });

app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR

detector = ModelLoader.load_detector()
embedder = ModelLoader.load_embedder()
manager =ImageEmbeddingManager();
helper = ImageHelper(detector, embedder,manager, UPLOAD_FOLDER, STATIC_FOLDER)


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
