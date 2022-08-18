import sys, os, torch, cv2
from pathlib import Path

BASE_DIR = Path(os.getcwd())
sys.path.append(str(BASE_DIR))

from glob import glob
from extractors.pose_extractors.pose_extractor import PoseEmbeddingExtractor
from tqdm import tqdm


def remove_non_poses(input_dir, split):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pee = PoseEmbeddingExtractor(device=device)
    file_name = f"pose_error_{split}.txt"
    os.makedirs("./data/error_indexes", exist_ok=True)
    non_pose_files = open(f"./data/error_indexes/{file_name}", "w")
    img_pattern = os.path.join(input_dir, "*.jpg")
    images = glob(img_pattern)
    for image_path in tqdm(images):
        try:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            pee.extract_embedding(img)

        except Exception as e:
            print(e)
            img_id = image_path.split("/")[-1].split(".")[0]
            non_pose_files.write(f"{img_id}")
            non_pose_files.write(os.linesep)


if __name__ == "__main__":
    split = "val"
    input_dir = BASE_DIR / "data/images/" / split
    remove_non_poses(input_dir, split)
