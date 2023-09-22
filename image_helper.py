import numpy as np
from insightface.utils.face_align import norm_crop
import os
import cv2
from PIL import Image

class ImageHelper:
    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    }

    # Load model on startup
    def __init__(self, detector,embedder, UPLOAD_FOLDER, STATIC_FOLDER):
        self.detector = detector
        self.embedder = embedder
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER

    @staticmethod
    def calculate_similarity(emb_a, emb_b):
        similarity = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        )
        return similarity

    def __align_single_image(self, face, selected_face, filename, img):
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"
        aligned_path = os.path.join(self.STATIC_FOLDER, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename

    def detect_faces_in_image(self, filename, images):
        img, faces = self.__extract_faces(filename)
        if faces:
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
            detected_path = os.path.join(self.STATIC_FOLDER, detected_filename)
            # message += f"path {detected_path}. "
            cv2.imwrite(detected_path, img)
            images.append(detected_filename)
            return len(faces)
        else:
            images.append(filename)

    def create_aligned_images(self, filename, images):
        img, faces = self.__extract_faces(filename)
        face_count = 0

        for face in faces:
            aligned_filename = self.__align_single_image(
                face, face_count, filename, img
            )
            images.append(aligned_filename)
            face_count += 1
        return face_count

    def __extract_faces(self, filename):
        path = os.path.join(self.UPLOAD_FOLDER, filename)
        img = cv2.imread(path)
        faces = self.detector.get(img)
        return img, faces

    @staticmethod
    def extract_embedding(face_data):
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
        

    def generate_embedding(self,path,selected_face):
        errors=[];
        embedding=None;
        if self.embedder:
            img = cv2.imread(path)
            faces = self.embedder.get(img)
            if faces:
                if selected_face == -2:
                    embedding = ImageHelper.extract_embedding(faces[0])
                elif len(faces) == 1:
                    embedding = ImageHelper.extract_embedding(faces[0])
                else:
                    embedding = ImageHelper.extract_embedding(
                        faces[selected_face]
                    )
               
            else:
                print("No faces detected.")  # Debug log
                errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embedding,errors;

    def get_most_similar_image(self,selected_face,filename):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        user_embedding,temp_err=self.generate_embedding(user_image_path,selected_face);
        errors=errors+temp_err
        max_similarity = -1
        most_similar_face_num = None
        with os.scandir(self.UPLOAD_FOLDER) as entries:
            for entry in entries:
                if (
                    entry.name != filename
                    and filename not in entry.name
                ):
                    if entry.is_file() and ImageHelper.allowed_file(entry.name):
                        with Image.open(entry.path) as img:
                            img = cv2.imread(entry.path)
                            faces = self.embedder.get(img)
                            if faces:
                                for i in range(len(faces)):
                                    embedding = (ImageHelper.extract_embedding(faces[i]))
                                    if embedding is not None:
                                        similarity = ImageHelper.calculate_similarity(
                                            user_embedding, embedding
                                        )
                                        if similarity > max_similarity:
                                            max_similarity = similarity
                                            most_similar_image = entry.name
                                            most_similar_face_num =i;
        return most_similar_image,most_similar_face_num,max_similarity,errors;

    @staticmethod
    def allowed_file(filename):
        extension=os.path.splitext(filename)[1];
        return extension.lower() in ImageHelper.ALLOWED_EXTENSIONS;
