from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import os
import json
import util
import tempfile
import traceback;
from image_helper import ImageHelper
from image_embedding_manager import ImageEmbeddingManager
from image_group_repository import ImageGroupRepository
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
cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins


@app.route("/pool/<path:filename>")
@cross_origin()
def custom_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static_images/<path:filename>")
@cross_origin()
def processed_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


@app.route("/api/upload", methods=["POST"])
def upload_image():
    images = []
    errors = []
    files = request.files.items()
    faces_length = [0] * len(request.files)
    current_images = []
    for image_name, file in files:
        if file and file.filename:
            filename = file.filename.replace("_", "")
            # if ImageHelper.allowed_file(filename):
            session.pop("uploaded_images", None)
            path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                file.save(path)
                # Generate the embeddings for all faces and store them for future indexing
                _, temp_err = helper.generate_all_emb(filename)
                errors = errors + temp_err

                if len(temp_err) > 0:
                    os.remove(path)
                    current_images.append(None)
                else:
                    current_images.append(filename)
                    images.append(filename)
            except Exception as e:
                tb=traceback.format_exc();
                errors.append(f"Failed to save {filename} due to error: {str(e)}")

            # else:
                # errors.append(f"Invalid file format for {filename}. ")

        # if(len(errors)==0):
        # Detect faces immediately after uploading to fill the combo box
        for i in range(len(current_images)):
            if current_images[i]:
                faces_length[i] = helper.create_aligned_images(
                    current_images[i], images
                )
        manager.save()
    return jsonify({"images": images, "faces_length": faces_length, "errors": errors})


@app.route("/api/delete", methods=["GET"])
def delete_embeddings():
    manager.delete()
    return jsonify({"result": "success"})


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
            faces_length[i] = face_count
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
            face_count, _ = helper.detect_faces_in_image(filename, images)
            faces_length[i] = face_count
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
    uploaded_images = request.form.getlist("images")
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding = manager.get_embedding_by_name(filename)
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                # Generate an embedding for a specific face(first by default) in each image
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], combochanges[i]
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
        similarity = util.calculate_similarity(embeddings[0], embeddings[1])
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
        }
    )


@app.route("/api/find", methods=["POST"])
@cross_origin()
def find_face_in_image():
    filename = request.form.get("image")
    faces_length = [0]
    messages = []
    errors = []
    boxes = []
    if "aligned" in filename or "detected" in filename:
        path = os.path.join(STATIC_FOLDER, filename)
    else:
        path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        face_count, boxes = helper.detect_faces_in_image(filename, [])
        faces_length = face_count
        messages.append(f"{face_count} detected faces in {filename}.")
    else:
        errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "boxes": boxes,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/filter",methods=["GET"])
def filter():
    threshold = float(request.form.get("threshold", 999))
    if threshold<1:
        deleted=helper.filter(threshold)
        return jsonify({"success":"true","deleted":deleted})
    return jsonify({"success":"false","error":"threshold must be below 1"})

@app.route("/api/ping",methods=["GET"])
def ping():
    return jsonify({"response":"pong"});



@app.route("/api/cluster", methods=["POST"])
def get_groups():
    jsonData=request.get_data();
    data=json.loads(jsonData) if jsonData else {};
    eps=float(data["max_distance"]) if "max_distance" in data else 0.5; 
    min_samples=int(data["min_samples"]) if "min_samples" in data else 4; 
    retrain=data["retrain"] if "retrain" in data else False; 
    value_groups=helper.cluster_images(eps,min_samples);
    if(retrain):
        groups.train_index(value_groups);
        groups.save_index();
        return jsonify(value_groups);
    modified_group={};
    index=groups.index;
    for cluster_id,images in value_groups.items():
        for image in images:
            group_name=cluster_id;
            if(image in index):
                group_name=index[image];
            if(group_name in modified_group):
                modified_group[group_name].append(image);
            else:
                modified_group[group_name]=[image];
    return jsonify(modified_group);

@app.route("/api/video",methods=["POST"])
def process_video():
    pass;


@app.route("/api/change_group_name", methods=["POST"])
def change_group_name():
    data=json.loads(request.get_data());
    old= data['old'];
    new= data['new'];
    if(old and new and old.strip() and new.strip()):
        groups.change_group_name(old,new);
    return jsonify(success=True);
@app.route("/api/check_many", methods=["POST"])
def find_similar_images():
    errors = []
    images = []
    files = request.files.items()
    k = int(request.form.get("number_of_images", 5))
    with tempfile.TemporaryDirectory() as temp_dir:
        for image_name, file in files:
            if file and file.filename:
                filename = file.filename.replace("_", "")
                if ImageHelper.allowed_file(filename):
                    path = os.path.join(temp_dir, filename)
                    try:
                        file.save(path)
                        # Generate the embeddings for all faces and store them for future indexing
                        embs, temp_err = helper.generate_all_emb(filename, False)
                        similar = helper.get_similar_images(embs[0], filename, k)
                        for x in similar:
                            sim_emb = manager.get_embedding(x["index"])
                            similarity =  util.calculate_similarity(
                                embs[0], sim_emb
                            )
                            images.append(
                                {"name": x["name"], "similarity": float(similarity)}
                            )

                        errors = errors + temp_err

                    except Exception as e:
                       errors.append(
                            f"Failed to save {filename} due to error: {str(e)}"
                        )

                else:
                    errors.append(f"Invalid file format for {filename}. ")
    return jsonify({"images": images, "errors": errors})


@app.route("/api/check", methods=["POST"])
def find_similar_image():
    most_similar_image = None
    messages = []
    errors = []
    similarity = -1
    face_length = 0
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    if len(current_image) > 0:
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
            "image": most_similar_image,
            "face": most_similar_face_num,
            "face_length": face_length,
            "errors": errors,
            "messages": messages,
        }
    )


app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR

detector = ModelLoader.load_detector(1024)
detector_zoomed = ModelLoader.load_detector(160)
embedder = ModelLoader.load_embedder()
manager = ImageEmbeddingManager();
groups = ImageGroupRepository();
helper = ImageHelper(
    detector, detector_zoomed, embedder,groups, manager, UPLOAD_FOLDER, STATIC_FOLDER
)
manager.load()

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
