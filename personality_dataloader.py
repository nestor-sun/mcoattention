import torch
from torch.utils.data import Dataset
import json
import numpy as np

class FirstImpressions(Dataset):
    def __init__(self, img_path, audio_path, text_path, label_file):
        self.img_path = img_path
        self.audio_path = audio_path
        self.text_path = text_path
        self.label_dict = json.load(open(label_file, 'r'))
        self.id_list = list(self.label_dict.keys())

    def __getitem__(self, index):
        video_id = self.id_list[index]
        img = np.load(self.img_path + video_id + '.npy')
        mfcc = np.load(self.audio_path + video_id + '.npy')
        text = np.load(self.text_path + video_id + '.npy')

        mfcc = torch.from_numpy(mfcc).float()
        img = torch.from_numpy(img).float()
        text = torch.from_numpy(text).float()
        return mfcc, img, text, torch.tensor(self.label_dict[video_id]).float()

    def __len__(self):
        return len(self.id_list)

