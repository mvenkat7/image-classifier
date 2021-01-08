Machine Learning Engineer Nanodegree - Deep Learning
Project: Flower Image Classifier
This project is part of the Udacity Machine Learning Engineer Nanodegree. The goal is to create an image classifier to identify flowers. The project consists of 2 parts:

Implementation of training and prediction in Jupyter notebook
Porting the code from Jupyter notebook to 2 separate scripts: to train a network model and to classify an image based on an existing trained network model.
Install
This project requires Python 3.x and the following Python libraries installed:

NumPy
Pandas
matplotlib
pytorch
You will also need to have software installed to run and execute an iPython Notebook

The easiest way to set up the envirnment is using pipenv. The repository contain a Pipfile that installs all dependencies and Python modules.

$ pipenv install
Installing dependencies from Pipfile.lock (8732c5)...
...
Code
For the 1st part, the code is provided in the Image Classifier Project.ipynb notebook file.

For the 2nd part, the code is in the train.py and predict.py file. In addition to these, you'll also need the files model.py and utils.py.

Run
In a terminal or command window, navigate to the top-level project directory (that contains this README) and run the following command:

pipenv run jupyter notebook Image\ Classifier\ Project.ipynb
This will open the iPython Notebook software and project file in your browser.

For part 2, you can execute the python scripts as illustrated below

To train a new network model

$ pipenv run ./train.py --help
usage: train.py [-h] [--save_directory SAVE_DIRECTORY] [--arch ARCH]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu]
                data_directory

positional arguments:
  data_directory

optional arguments:
  -h, --help            show this help message and exit
  --save_directory SAVE_DIRECTORY
  --arch ARCH
  --learning_rate LEARNING_RATE
  --hidden_units HIDDEN_UNITS
  --epochs EPOCHS
  --gpu

 $ pipenv run ./train.py ./flowers/ --save_directory models --learning_rate 0.01
 Epoch 1/20..
 ...
To make a prediction using an training model

$ pipenv run ./predict.py --help
usage: predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu]
                  image_path checkpoint_path

positional arguments:
  image_path
  checkpoint_path

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K
  --category_names CATEGORY_NAMES
  --gpu

$ pipenv run ./predict.py ./flowers/train/70/image_05278.jpg ./model_checkpoint.pth --top_k 3
Predicting flower class for image ./flowers/train/70/image_05278.jpg ..
Prediction 1: tree poppy .. (59.028%)
Prediction 2: pelargonium .. (34.462%)
Prediction 3: japanese anemone .. (1.529%)
Data
The dataset consists of about 8200 images of flowers divided over 102 different categories. The description of the dataset is available here. It can be downloaded from here