# MultiModalSentimentAnalysis

This is the main project of the course Natural Langugae Processing course at Sharif University of Technology, Spring 2022. We used pretrained networks of different tasks including Face Emotion Detection, Text sentiment analysis, Pose Recognition, Scene Recognition ... to extract embeddings (features) of a pair of image and text. We then concatenated 
all embeddings and traind a simple fully connected network on top of the concatenated feature vector. Our multi modal approach increased f1 by ~0.15 for sentiment anlysis task with respect to plain text sentiment models. For more information
refer to our doc located in LaTeX folder.

## Running instructions:
#### Running stand alone with notebook (or Colab) 
1. Use Full.ipynb
2. Initialize your data directory. For more instructions check data/README.md


#### Running locally
1. Install requirements.txt
2. Run preprocess/save_embeddings for test,train and val data
3. Run with train.py


## Project structure

Project is organized as follow:


- data/

- deployment/

- extractors/

  - face_extractor
  - pose_extractor
  - scene_extractor
  - text_extractor
  
- LaTeX/

- preprocess/

- results/


**data** folder is specificly addressed in it's own README.md.

**deployment** code is used for deploying project on ray server.

**extractors** are classes used for extracting different embeddings from the image or text. Each embedding extractor and it's related files are in a different folder. Text extractors are for two different languages, Gnglish and German.

**LaTeX** contains project doc and it's latex file and images.

**preprocess** contains our custom dataset, along with save_embeddings.py. The latter is used for extracting all embeddings and saving them in the data/saved_features directory. This step is obligatory before running main.py.

**results** are notebook to calculate single modal results for comparing with out multimodal network.