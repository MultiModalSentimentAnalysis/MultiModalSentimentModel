import sys, os
from pathlib import Path

BASE_DIR = Path(os.getcwd())
sys.path.append(str(BASE_DIR))

from glob import glob
from extractors.pose_extractors.pose_extractor import PoseEmbeddingExtractor
from tqdm import tqdm


def remove_non_poses(input_dir, split):

    pee = PoseEmbeddingExtractor()
    file_name = f"pose_error_{split}.txt"
    os.makedirs("./data/error_indexes", exist_ok=True)
    non_pose_files = open(f"./data/error_indexes/{file_name}", "w")
    img_pattern = os.path.join(input_dir, "*.jpg")
    images = glob(img_pattern)
    for img in tqdm(images):
        try:
            pee.extract_embedding(img)

        except Exception as e:
            img_id = img.split("/")[-1].split(".")[0]
            non_pose_files.write(f"{img_id}")
            non_pose_files.write(os.linesep)


if __name__ == "__main__":
    split = "test"
    input_dir = BASE_DIR / "data/images/" / split
    remove_non_poses(input_dir, split)
