# HieRec
- Code of our HieRec model

# Data Preparation
- If you want to try this project, you should download MIND-Small dataset in https://msnews.github.io/index.html
- All data in MIND are stored in data-root-path
- We used the glove.840B.300d embedding vecrors in https://nlp.stanford.edu/projects/glove/
- The embedding file should be stored in embedding\_path\glove.840B.300d.txt
- The meta data of entity (including news entities, and pre-trained transE embeddings) should be stored in KG\_root\_path

# Code Files
- preprocess.py: containing functions to preprocess data
- utils.py: containing some util functions
- models.py: containing codes for implementing the KIM model
- hypers.py: containing settings of hyper-parameters
- Main.ipynb: containing codes for model training and evaluation
- ProcessRawData.ipynb: using for converting raw MIND files (train/news.tsv, train/behaviors.tsv, dev/news.tsv, dev/behaviors.tsv), into files used in our codes (data_root_path/docs.tsv, train.tsv, test.tsv)

