import pickle
import faiss
import os
import numpy as np
import math
class ImageEmbeddingManager:
    def __init__(self):
        self.db_embeddings={"names":[],"embeddings":np.empty((0, 512), dtype='float32')};
    PKL_path='static/embeddings.pkl'
    def train_IVFPQ_index(self,data):
        nlist = 100;
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        m = 16
        nbits = int(math.floor(np.log2(len(data))))
        quantizer = faiss.IndexFlatIP(d);
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        
        self.index.train(data);
        self.index.add(data);
    def train_HNSW_index(self,data):
        #M is the amount of connection of each node(datapoint)
        M=int(np.log2(len(data)).round())
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        self.index = faiss.IndexHNSWFlat( d,M);
        self.index.add(data);
    
    def search(self,embedding,k):
        data=self.db_embeddings["embeddings"];
        # Define the number of clusters (nlist) for the IVFPQ index
        threshold = 25600;
        if len(data)>=threshold:
            # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
            self.train_IVFPQ_index(data);
        else:
            self.index = faiss.IndexFlatIP(512);
            self.index.add(data)
        return self.find_closest_vector(embedding,k);
        
    def delete(self):
        self.db_embeddings={"names":[],"embeddings":np.empty((0, 512), dtype='float32')};
        if os.path.exists(ImageEmbeddingManager.PKL_path):
            os.remove(ImageEmbeddingManager.PKL_path);
    def save(self):
        with open(ImageEmbeddingManager.PKL_path, 'wb') as file:
            pickle.dump(self.db_embeddings, file)
    def load(self):
        if os.path.exists(ImageEmbeddingManager.PKL_path):
            with open(ImageEmbeddingManager.PKL_path, 'rb') as file:
                self.db_embeddings = pickle.load(file)

    def find_closest_vector(self,new_vector,k):

        distances,indexes = self.index.search(new_vector, k)
        # return indexes based on distance
        # Create a list of objects
        result = []

        for i in range(len(distances[0])):
            obj = {'index': indexes[0][i], 'distance': distances[0][i]}
            result.append(obj)
        return result;

    def remove_embedding_by_name(self,name):
        index=self.get_index_by_name(name);
        self.remove_embedding_by_index(index);
    def remove_embedding_by_index(self,index):
        if(index>-1):
            self.db_embeddings["names"].pop(index);
            self.db_embeddings["embeddings"]=np.delete(self.db_embeddings["embeddings"],index,axis=0);
    def add_embedding(self,embedding,name):
        existing=self.get_embedding_by_name(name);
        if(len(existing)==0):
            np_emb=np.array(embedding).reshape(1,-1);
            self.db_embeddings["names"].append(name);
            self.db_embeddings["embeddings"]=np.vstack(
            (self.db_embeddings["embeddings"],np_emb));
    def get_name(self,idx):
        if(len(self.db_embeddings["names"])>0):
            return self.db_embeddings["names"][idx];
        return "";
    def get_embedding(self,idx):
        if(len(self.db_embeddings["embeddings"])>0):
            return self.db_embeddings["embeddings"][idx];
        return [];
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

