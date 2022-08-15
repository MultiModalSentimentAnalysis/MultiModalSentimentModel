import os
import pandas as pd
import numpy as np
import cv2
import torch 
from torch.utils.data import Dataset, DataLoader
import ast

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, base_path, dataset_type, tokenizer=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tokenizer = tokenizer
        self.text_path = os.path.join(base_path, f"english_{dataset_type}.txt")
        self.image_index_path = os.path.join(base_path, f"image_index_{dataset_type}.txt")
        self.sentiment_path = os.path.join(base_path, f"sentiment_{dataset_type}.txt")
        self.image_dir = os.path.join(base_path, dataset_type)
        self.data_info = self.read_info()
        self.image_pad = 10

    
    def read_info(self):
        with open(self.text_path) as f:
            texts = [t.strip() for t in f.readlines()]
        with open(self.image_index_path) as f:
            images = [ast.literal_eval(t.strip()) for t in f.readlines()]

        with open(self.sentiment_path) as f:
            sentiments = [int(t.strip()) for t in f.readlines()]
        df = pd.DataFrame([texts, images, sentiments], index=["text", "images", "sentiment"]).transpose()
        return df


    def __len__(self):
        return self.data_info.shape[0]
    

    def get_image_features(self, image_indicies):
        if image_indicies: # do the work if image indices is not none!
            img_paths = img_paths[:self.image_pad] # only keeping images within the pad. should change to better selection
            images = list()
            for img_id in img_paths:
                img_name = os.path.join(self.root_dir,
                                    f"{img_id}.jpg")
                image = cv2.imread(img_name)
                images.append(image)
            padding = self.image_pad - len(images)
            for pad in range(0, padding):
                images.append(np.zeros(images[0].shape))
            return images
        else:
            images = list()
            for pad in range(0, padding):
                images.append(np.zeros((256, 256, 3)))
            return images

    def get_sentiment(self, sentiment):
        return sentiment

    def get_text_features(self, text):
        if self.tokenizer:
            return self.tokenizer.encode(text)    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data_info.iloc[idx]
        img_paths = data["images"]
        images = self.get_image_features(img_paths)
        sentiment = self.get_sentiment(data["sentiment"])
        text_embed = self.get_text_features(data["text"])
        sample = {'images': images,
                  "text": text_embed,
                  "sentiment": sentiment}

        return sample
