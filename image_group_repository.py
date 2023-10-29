import pickle
import os;
class ImageGroupRepository:
    def __init__(self):
        self.load_index();
        
    PKL_path='static/groups.pkl'
 
    def train_index(self,data:dict):
        for key,images in data.items():
            for image in images:
                self.index[image]=str(key)
    def save_index(self):
        with open(ImageGroupRepository.PKL_path, 'wb') as file:
            pickle.dump(self.index, file)
    def load_index(self):
        if os.path.exists(ImageGroupRepository.PKL_path):
            with open(ImageGroupRepository.PKL_path, 'rb') as file:
                self.index = pickle.load(file)
        else:
            self.index={}

    def delete_index(self):
        self.index={};
        if os.path.exists(ImageGroupRepository.PKL_path):
            os.remove(ImageGroupRepository.PKL_path);

    def change_group_name(self, old_id, new_id):
        for image_name,group_id in self.index.items():
                if(self.index[image_name]==old_id):
                    self.index[image_name]=new_id 
    def get_id_groups(self):
        id_groups={};
        for image_name,group_id in self.index.items():
            if group_id not in id_groups:
                id_groups[group_id] = []
            else:
                id_groups[group_id].append(image_name)
        return id_groups;
