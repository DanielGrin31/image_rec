import faiss
import numpy as np
import math
class ImageEmbeddingManager:
    def __init__(self):
        self.db_embeddings={"names":[],"embeddings":np.empty((0, 512), dtype='float32')};

    def train_IVFPQ_index(self,data):
        nlist = 32;
        d=512;
        # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
        m = 8
        nbits = int(math.floor(np.log2(len(data))))
        quantizer = faiss.IndexFlatL2(d);
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
    
    def search(self,embedding):
        data=self.db_embeddings["embeddings"];
        # Define the number of clusters (nlist) for the IVFPQ index
        threshold = 100;
        if len(data)>=threshold:
            # Define the number of subquantizers (m) and number of bits per subquantizer (nbits)
            self.train_IVFPQ_index(data);
        else:
            self.index = faiss.IndexFlatL2(512);
            self.index.add(data)
        return self.find_closest_vector(embedding);
        


    def find_closest_vector(self,new_vector,k=5):

        closest_idx = self.index.search(new_vector, k)
        # return indexes based on distance
        return closest_idx[1][0];
    
    def add_embedding(self,embedding,name):
        existing=self.get_embedding_by_name(name);
        if(len(existing)==0):
            np_emb=np.array(embedding).reshape(1,-1);
            self.db_embeddings["names"].append(name);
            self.db_embeddings["embeddings"]=np.vstack(
            (self.db_embeddings["embeddings"],np_emb));


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