from glob import glob
import os
import matplotlib.pyplot as plt
import random

import numpy as np
import torch
import random

from image_analysis.face_detection import FaceDetection
from image_analysis.face_alignment import FaceAlignment
from image_analysis.face_normalization import FaceNormalizer
from image_analysis.face_emotion_recognition import FaceEmotionRecognizer

from image_analysis.representation_extraction import EmotionRepresentationExtractor

# fd = FaceDetection("MTCNN", minimum_confidence=0.95)
# fa = FaceAlignment()
# fn = FaceNormalizer()

# model_name = "enet_b0_8_best_afew"
# # model_name = 'enet_b0_8_best_vgaf'
# # model_name='enet_b0_8_va_mtl'
# # model_name='enet_b2_8'

# fer = FaceEmotionRecognizer(model_name, device="cpu")

# fre = (
#     EmotionRepresentationExtractor()
#     .set_face_detection_model(
#         fd,
#     )
#     .set_face_alignment_model(
#         fa,
#     )
#     .set_face_normalizer_model(fn)
#     .set_face_emotion_recognition_model(fer)
# )
# non_face_files = open("/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/logs/face_error.txt", "w")
# image_paths = "/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/*/*.jpg"
# images = glob(image_paths)
# os.makedirs("/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/dev/trash", exist_ok=True)
# os.makedirs("/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/test/trash", exist_ok=True)
# os.makedirs("/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/train/trash", exist_ok=True)
# for img in images:
#     try:
#         image = plt.imread(img)
#         fre.extract_representation(image)
#     except Exception as e:
#         img_dirs = img.split("/")

#         # logging 
#         non_face_files.write(f"image from set: {img_dirs[-2]} with id: {img_dirs[-1]} doesn't contain a face. moving to trash file")
#         non_face_files.write(os.linesep)

#         # moving data
#         new_img_dir = os.path.join(*img_dirs[:-1], "trash", img_dirs[-1])
#         # print("new image dir: ", new_img_dir)
#         # os.makedirs(new_img_dir, exist_ok=True)
#         os.rename(img, new_img_dir)

def remove_non_spaces(input_dir, output_dir):
    fd = FaceDetection("MTCNN", minimum_confidence=0.95)
    fa = FaceAlignment()
    fn = FaceNormalizer()

    model_name = "enet_b0_8_best_afew"
    # model_name = 'enet_b0_8_best_vgaf'
    # model_name='enet_b0_8_va_mtl'
    # model_name='enet_b2_8'

    fer = FaceEmotionRecognizer(model_name, device="cpu")

    fre = (
        EmotionRepresentationExtractor()
        .set_face_detection_model(
            fd,
        )
        .set_face_alignment_model(
            fa,
        )
        .set_face_normalizer_model(fn)
        .set_face_emotion_recognition_model(fer)
    )
    os.makedirs("./logs", exist_ok=True)
    non_face_files = open("./logs/face_error.txt", "w")
    img_pattern = os.path.join(input_dir, "*.jpg")
    images = glob(img_pattern)
    os.makedirs(output_dir, exist_ok=True)
    for img in images:
        try:
            image = plt.imread(img)
            fre.extract_representation(image)
        except Exception as e:
            img_dirs = img.split("/")

            # logging 
            non_face_files.write(f"image from set: {img_dirs[-2]} with id: {img_dirs[-1]} doesn't contain a face. moving to trash file")
            non_face_files.write(os.linesep)

            # moving data
            new_img_dir = os.path.join(output_dir, img_dirs[-1])
            # print("new image dir: ", new_img_dir)
            # os.makedirs(new_img_dir, exist_ok=True)
            os.rename(img, new_img_dir)

if __name__ == "__main__":
    input_dir = "/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/dev"
    output_dir = "/home/sahel/personal/university/NLP/project/MultiModalEmotionRecognition/data/images/dev/trash"
    remove_non_spaces(input_dir, output_dir)