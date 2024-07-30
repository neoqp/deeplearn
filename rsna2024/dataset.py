import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import torchvision
import pandas as pd
import os
import pydicom

class DCMImageDataset(Dataset):
    def __init__(self, series, coordinates_file, descriptions_file, train_file, img_dir, file_counts=None):
        self.coordinates = coordinates_file
        self.descriptions =  descriptions_file
        self.train = train_file
        self.series = series
        self.img_dir = img_dir

        if file_counts:
            self.file_counts = file_counts
        
        else:
            self.file_counts = {}
            study_ids = os.listdir(self.img_dir + 'train_images')

            for study_id in study_ids:
                series_ids = os.listdir(self.img_dir + 'train_images/' + study_id)
                tmp = {}
                for series_id in series_ids:
                    tmp[series_id] = os.listdir(self.img_dir + 'train_images/' + study_id + '/' + series_id)

                self.file_counts[study_id] = tmp


        merge = descriptions_file.merge(train_file, on='study_id', how='left')
        f = merge[merge['series_description'] == series]
        result = []
        for i in range(len(f)):
            study_id = f.iloc[i]['study_id']
            series_id = f.iloc[i]['series_id']
            ndf = f[(f['study_id'] == study_id) & (f['series_id'] == series_id)]
            
            expanded_dfs = []
            for j in self.file_counts[str(study_id)][str(series_id)]:
                ndf['number'] = j.split('.')[0]
                expanded_dfs.append(ndf.copy())
            
            dfs = pd.concat(expanded_dfs).reset_index(drop=True)
            result.append(dfs.copy())

        mapping = {'Normal/Mild' : 0, 'Moderate' : 1, 'Severe' : 2}
        self.df = pd.concat(result).reset_index(drop=True)
        self.label_column = self.df.columns[3:-1]
        self.df[self.label_column] = self.df[self.label_column].replace(mapping)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        study_id = str(self.df.iloc[idx]['study_id'])
        series_id = str(self.df.iloc[idx]['series_id'])

        img_path = os.path.join(str(self.img_dir + 'train_images'), study_id)
        img_path = os.path.join(img_path, series_id)
        img_path = img_path + '/' + str(self.df.iloc[idx]['number']) + '.dcm'

        image = torch.from_numpy(pydicom.dcmread(str(img_path)).pixel_array.astype(np.float64))
        image = image.unsqueeze(0)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (224,224), mode='bilinear')
        image = image.reshape(224, 224)
        image = F.normalize(image)
        
        label = self.df.iloc[idx][self.label_column].tolist()
        label = torch.tensor(label)

        return image, label