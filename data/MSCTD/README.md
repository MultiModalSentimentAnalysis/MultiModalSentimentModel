# Data

Data folder is organized as follow:

data/
- images/
    - train/
    - test/
    - val/
- saved_features/
- texts/
  - english/
    - {train/test/val}.txt
  - german/
    - {train/test/val}.txt
- labels/
  - sentiment_{train/test/val}.txt



Each split image is located in images/split. For downloading images you can visit [MSCTD dataset git hub](https://github.com/XL2248/MSCTD/tree/main/MSCTD_data). Click on 
[train](https://drive.google.com/file/d/1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj/view?usp=sharing), 
[test](https://drive.google.com/file/d/1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W/view?usp=sharing) and 
[validation](https://drive.google.com/file/d/12HM8uVNjFg-HRZ15ADue4oLGFAYQwvTA/view?usp=sharing)
 for a direct link to data

saved_features is used for saving extracted embeddings. 

texts folder contains input texts (not their embeddings).

labels/ contains sentiment labels for image/text pairs.

  