import numpy as np
from insightface.utils.face_align import norm_crop
import os
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import util
import cv2
from PIL import Image
from image_embedding_manager import ImageEmbeddingManager
class ImageHelper:
    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }

    # Load model on startup
    def __init__(self, detector,detector_zoomed,embedder,groups,emb_manager, UPLOAD_FOLDER, STATIC_FOLDER):
        self.detector = detector
        self.detector_zoomed = detector_zoomed
        self.embedder = embedder
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER
        self.groups=groups;
        self.emb_manager=emb_manager;


    def __align_single_image(self, face, selected_face, filename, img):
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"
        aligned_path = os.path.join(self.STATIC_FOLDER, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename

    def detect_faces_in_image(self, filename, images):
        img, faces = self.__extract_faces(filename)
        boxes=[]
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
                box=face['bbox'].astype(int).tolist();
                boxes.append(box)
            detected_filename = "detected_" + filename
            detected_path = os.path.join(self.STATIC_FOLDER, detected_filename)
            # message += f"path {detected_path}. "
            cv2.imwrite(detected_path, img)
            images.append(detected_filename)
            return len(faces),boxes
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
        close_faces=self.detector_zoomed.get(img)
        far_faces=self.detector.get(img)
        faces=far_faces.copy();
        for j in range(len(close_faces)):
                    duplicate=False;
                    for far_face in far_faces:
                        if(util.are_bboxes_similar(close_faces[j]['bbox'],far_face['bbox'],20)):
                            duplicate=True;
                    if(not duplicate):
                        faces.append(close_faces[j])
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
        
    def generate_all_emb(self,filename,save=True):
        errors=[];
        embedding=None;
        embeddings=[];
        if self.embedder:
            img,faces=self.__extract_faces(filename);                
            if faces:
                for i in range(len(faces)):
                    embedding=self.embedder.get(img,faces[i]);
                    embeddings.append(np.array(embedding));
                    if(save):
                        self.emb_manager.add_embedding(embedding,f"aligned_{i}_{filename}");
            else:
                print("No faces detected.")  # Debug log
                errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embeddings,errors;
    def generate_embedding(self,filename,selected_face):
        errors=[];
        embedding=None;
        if self.embedder:
            img,faces=self.__extract_faces(filename)
            if faces:
                if selected_face == -2 or len(faces) == 1:
                    i=0
                else:
                    i=selected_face

                embedding = ImageHelper.extract_embedding(faces[i]);
            else:
                print("No faces detected.")  # Debug log
                errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embedding,errors;

    def get_similar_images(self,user_embedding,filename,k=5):
        np_emb=np.array(user_embedding).astype("float32").reshape(1,-1)
        result=self.emb_manager.search(np_emb,k);
        filtered=[]
        seen_distances=[]
        for r in result:
            if r["distance"] not in seen_distances:
                seen_distances.append(r["distance"])
                i=r["index"];
                name=self.emb_manager.get_name(i);
                if(name.split('_')[-1]!=filename):
                    filtered.append({"index":i,"name":name})
        valid=[x for x in filtered if len(emb := self.emb_manager.get_embedding(x['index']))>0 
                and not np.allclose(emb,user_embedding,rtol=1e-5,atol=1e-8)]
        return valid;
    def filter(self,threshold):
        manager=self.emb_manager
        errors=[]
        
        original_length=len(manager.db_embeddings["names"]);
        for name in manager.db_embeddings["names"]:
            embedding=manager.get_embedding_by_name(name)
            valid=self.get_similar_images(embedding,name.split('_')[-1]);
            for image in valid:            
                match=image['name'];
                _,facenum,filename=match.split('_');
                similarity=util.calculate_similarity(
                    self.emb_manager.get_embedding(image['index'])
                    ,embedding);
                if(similarity>threshold):
                    manager.remove_embedding_by_index(image['index']);
        filtered_length=len(manager.db_embeddings["names"]);
        return original_length-filtered_length;
    def get_most_similar_image(self,selected_face,filename):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        most_similar_image=None;
        most_similar_face=-2;
        max_similarity=-1;
        facenum=-2;
        aligned_filename=f"aligned_{0 if selected_face == -2 else selected_face}_{filename}";
        embedding=self.emb_manager.get_embedding_by_name(aligned_filename)
        if len(embedding)>0:
            user_embedding=embedding;
        else:
            user_embedding,temp_err=self.generate_embedding(filename,selected_face);
            errors=errors+temp_err;
        if len(errors)==0:
            valid=self.get_similar_images(user_embedding,filename);
            for image in valid:
                try:  
                    match=image['name'];
                    _,facenum,filename=match.split('_');
                    similarity=util.calculate_similarity(
                        self.emb_manager.get_embedding(image['index'])
                        ,user_embedding);
                    if(similarity>max_similarity):
                        max_similarity=similarity;
                        most_similar_image=filename;
                        most_similar_face=int(facenum);
                except Exception as e:
                    print(f"failed to match image {match} because:\n{e}");
            if len(valid)==0:
                errors.append("No unique matching faces found!");
        else:
            errors=errors+temp_err;
        return most_similar_image,most_similar_face,max_similarity,errors;
    @staticmethod
    def allowed_file(filename):
        extension=os.path.splitext(filename)[1];
        return extension.lower() in ImageHelper.ALLOWED_EXTENSIONS;
    

    def cluster_images(self,max_distance,min_samples):
        # Assuming 'embeddings' is a list of your 512-dimensional embeddings
        similarity_matrix = cosine_similarity(self.emb_manager.db_embeddings["embeddings"])
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN

        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        unique_values = np.unique(labels)
        index_groups = {value: np.where(labels == value)[0] for value in unique_values}
        value_groups = {
            int(key): [self.emb_manager.db_embeddings["names"][index] for index in indexes]
            for key, indexes in index_groups.items()
        }
        return value_groups;