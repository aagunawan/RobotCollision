from math import floor
import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # self.data = np.genfromtxt('saved/submission.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the datasety
        pass
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        
        x = (self.normalized_data[idx][:-1]).astype(np.float32)
        y = (self.normalized_data[idx][-1]).astype(np.float32)

        return {'input': x, 'label': y}
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders():
    def __init__(self, batch_size):

        self.nav_dataset = Nav_Dataset()
        test_fraction = 0.3 # percent of data allocated for test
        data_length = len(self.nav_dataset)
        
        num_data_test = floor(test_fraction*data_length)
        num_data_train = data_length - num_data_test
        num_iter_test = floor(num_data_test /batch_size) # number iterations given batch size 
        num_iter_train = floor(num_data_train /batch_size) # number iterations given batch size 

        self.train_loader = [{}]*num_iter_train
        self.test_loader = [{}]*num_iter_test

        test_loader_temp = []
        train_loader_temp = []
        test_loader_label_temp = []
        train_loader_label_temp = []

        count_collision = 0 # number of samples that have collision
        for idx, sample in enumerate(self.nav_dataset):
            if self.nav_dataset[idx]['label'] == 1.0:
                count_collision += 1
        
        test_collision_fraction = 0.2
        num_collision_test = floor((test_collision_fraction* count_collision)) # number of collision in test total
        num_collision_train = floor((1-test_collision_fraction)* count_collision) # number of collision in test total
        num_non_collision_test = num_data_test - num_collision_test
        num_col_per_batch_test = floor(num_collision_test/num_iter_test)
        num_col_per_batch_train = floor(num_collision_train/num_iter_train)
        test_collision_counter = 0
        test_non_collision_counter = 0
        
        for idx, sample in enumerate(self.nav_dataset):
            
            if self.nav_dataset[idx]['label']== 1.0:               
                if test_collision_counter < num_collision_test:                  
                    test_loader_temp.append(self.nav_dataset[idx]['input'].tolist())
                    test_loader_label_temp.append(self.nav_dataset[idx]['label'].tolist())
                    test_collision_counter += 1              
                else:
                    train_loader_temp.append(self.nav_dataset[idx]['input'].tolist())
                    train_loader_label_temp.append(self.nav_dataset[idx]['label'].tolist())
              
            else:
                if test_non_collision_counter < num_non_collision_test:
                    test_loader_temp.append(self.nav_dataset[idx]['input'].tolist())
                    test_loader_label_temp.append(self.nav_dataset[idx]['label'].tolist())
                    test_non_collision_counter += 1              
                else:
                    train_loader_temp.append(self.nav_dataset[idx]['input'].tolist())
                    train_loader_label_temp.append(self.nav_dataset[idx]['label'].tolist())

        for id in range(num_iter_test):
            idx_test = 0    
            temp = []     
            temp_label =[] 
            while idx_test < batch_size:
                index = batch_size*id + idx_test
                temp.append(test_loader_temp[index])
                temp_label.append(test_loader_label_temp[index])
                idx_test += 1

            self.test_loader[id]['input'] = temp
            self.test_loader[id]['label'] = temp_label
        
        for id in range(num_iter_train):
            idx_train = 0    
            temp = []     
            temp_label =[] 
            while idx_train < batch_size:
                index = batch_size*id + idx_train
                temp.append(train_loader_temp[index])
                temp_label.append(train_loader_label_temp[index])
                idx_train += 1
            self.train_loader[id]['input'] = temp
            self.train_loader[id]['label'] = temp_label
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
