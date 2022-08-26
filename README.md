# MultiModalSentimentAnalysis

This is the main project of the course Natural Langugae Processing course at Sharif University of Technology, Spring 2022. We used pretrained networks of different tasks including Face Emotion Detection, Text sentiment analysis, Pose Recognition, Scene Recognition ... to extract embeddings (features) of a pair of image and text. We then concatenated 
all embeddings and traind a simple fully connected network on top of the concatenated feature vector. Our multi modal approach increased f1 by ~0.15 for sentiment anlysis task with respect to plain text sentiment models. For more information
refer to our doc located in LaTeX folder.


--------------------------------
## Running instructions
There are different ways to run the project. In this section we will introduce these ways and how to use them. 

### Running stand alone with notebook (or Colab) 
you can use Full.ipynb to with all the code on Google Colab instance or as a jupyter notebook. make sure your data directory follows instruction in data/README.md. 

Global settings are saved in settings.py. you can change these variables based on your need. 

### Running locally
you can run different steps of our project with python files with following steps. 

1. create a python virtual env and activate it

    `python3 -m venv .venv`

    `source .venv/bin/activate`
1. Install requirements.txt with pip 

    `pip install -r requirements.txt`
2. Run preprocess/save_embeddings for test,train and val data

   `python preprocess/save_embeddings.py`

3. train the model!

    `python train.py`

    You can also change the default training variables set in train.py file based on your specific need. 

---------------------------

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


**data**

 contains data used in this project. More instruction is addressed in it's own README.md.

**deployment** 

code is used for deploying project on ray server.

**extractors**

 are classes used for extracting different embeddings from the image or text. Each embedding extractor and it's related files are in a different folder. Text extractors are for two different languages, Gnglish and German.

**LaTeX** 

contains project's documentation and it's latex file and images.

**preprocess** 

contains our custom dataset, along with save_embeddings.py. The latter is used for extracting all embeddings and saving them in the data/saved_features directory. This step is obligatory before running main.py.

**results** 

are notebook to calculate single modal results for comparing with out multimodal network.