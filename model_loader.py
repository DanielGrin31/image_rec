from insightface.app import FaceAnalysis
import insightface
class ModelLoader:
    @staticmethod
    def load_detector(image_size=640):
        try:
            face_analyzer = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
            face_analyzer.prepare(ctx_id=1, det_thresh=0.75, det_size=(image_size,image_size))
            return face_analyzer

        except Exception as e:
            print("Error during model initialization:", e)
            return None

    @staticmethod
    def load_embedder(image_size=640):
        try:
            model_name = '/home/dangrin/.insightface/models/buffalo_l/w600k_r50.onnx' # Use the face recognition model
            embedder = insightface.model_zoo.get_model(model_name)
            embedder.prepare(ctx_id=1, det_thresh=0.5, det_size=(image_size, image_size))
            return embedder

        except Exception as e:
            print("Error during embedder model initialization:", e)
            return None

