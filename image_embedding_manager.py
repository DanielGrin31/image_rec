import faiss
import numpy as np
class ImageEmbeddingManager:
    def __init__(self):
        self.db_embeddings={"names":[],"embeddings":np.empty((0, 512), dtype='float32')};

    def search(self,embedding):
        # ImageHelper.db_embeddings["embeddings"] = np.unique(ImageHelper.db_embeddings["embeddings"], axis=0)
        index = faiss.IndexFlatL2(512)  # L2 distance is used for similarity search
        index.add(self.db_embeddings["embeddings"])
        return self.find_closest_vector(index,embedding);

    def find_closest_vector(self,index,new_vector,k=5):
        closest_idx = index.search(new_vector, k)
        return closest_idx[1][0];
    
    def add_embedding(self,embedding,name):
        existing=self.get_embedding_by_name(name);
        if(len(existing)==0):
            self.db_embeddings["names"].append(name);
            self.db_embeddings["embeddings"]=np.vstack(
            (self.db_embeddings["embeddings"],np.array(embedding).reshape(1,-1)));

    def get_name(self,idx):
        return self.db_embeddings["names"][idx];
    def get_embedding(self,idx):
        return self.db_embeddings["embeddings"][idx];
    def get_index_by_name(self,name):
        try:
            return self.db_embeddings["names"].index(name);
        except ValueError:
            return -1;
    def get_embedding_by_name(self,name):
        try:
            index=self.db_embeddings["names"].index(name);
            return self.db_embeddings["embeddings"][index];
        except ValueError:
            return [];