import os, cv2, torch, ast
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class MSCTDDataSet(Dataset):
    """MSCTD dataset."""

    def __init__(self, base_path="data/", dataset_type="train"):
        """
        Args:
            base_path (str): path to data folder
            dataset_type (str): dev, train, test
        """
        base_path = Path(base_path)
        self.text_path = base_path / f"english_{dataset_type}.txt"
        self.image_index_path = base_path / f"image_index_{dataset_type}.txt"
        self.sentiment_path = base_path / f"sentiment_{dataset_type}.txt"
        self.image_dir = base_path / "images" / dataset_type
        self.data_info = self.read_info()
        self.image_pad = 10

    def read_info(self):
        with open(self.text_path) as f:
            texts = [t.strip() for t in f.readlines()]
        with open(self.image_index_path) as f:
            images = [ast.literal_eval(t.strip()) for t in f.readlines()]

        with open(self.sentiment_path) as f:
            sentiments = [int(t.strip()) for t in f.readlines()]
        df = pd.DataFrame(
            [texts, images, sentiments], index=["text", "image", "sentiment"]
        ).transpose()
        return df

    def __len__(self):
        return self.data_info.shape[0]

    def get_image_features(self, image_indicies):
        if image_indicies:  # do the work if image indices is not none!
            # only keeping images within the pad. should change to better selection
            img_paths = img_paths[: self.image_pad]
            images = list()
            for img_id in img_paths:
                img_name = os.path.join(self.image_dir, f"{img_id}.jpg")
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

    def get_single_img_feature(self, idx):
        img_name = self.image_dir / f"{idx}.jpg"
        # image = np.load(img_name)
        image = cv2.imread(str(img_name))
        image.resize((495, 1024, 3))
        return image

    def get_sentiment(self, sentiment):
        return sentiment

    def get_text_features(self, text):
        return text

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data_info.iloc[idx]
        # img_paths = data["images"] # img indices seem wrong
        images = self.get_single_img_feature(idx)
        sentiment = self.get_sentiment(data["sentiment"])
        text_embed = self.get_text_features(data["text"])
        sample = {"images": images, "text": text_embed, "sentiment": sentiment}

        return sample


if __name__ == "__main__":
    dataset = MSCTDDataSet("data/", "val")
    print(dataset[0])
