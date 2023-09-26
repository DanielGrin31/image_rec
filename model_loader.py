from insightface.app import FaceAnalysis
class ModelLoader:
    @staticmethod
    def load_detector():
        try:
            face_analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
            face_analyzer.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
            return face_analyzer

        except Exception as e:
            print("Error during model initialization:", e)
            return None

    @staticmethod
    def load_embedder():
        try:
            model_name = "arcface_r100_v1"  # Use the face recognition model
            embedder = FaceAnalysis(model=model_name)
            embedder.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
            return embedder

        except Exception as e:
            print("Error during embedder model initialization:", e)
            return None

