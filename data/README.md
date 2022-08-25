# Data

Data folder is organized as follow:

data/
- images/
    - train/
    - test/
    - val/
- saved_features/
- texts/
  - english_{train/test/val}.txt
- labels/
  - sentiment_{train/test/val}.txt



Each split image is located in images/split. For downloading images you can visit:
https://github.com/XL2248/MSCTD/tree/main/MSCTD_data


saved_features is used for saving extracted embeddings. 

texts folder contains input texts (not their embeddings).

labels/ contains sentiment labels for image/text pairs.

  