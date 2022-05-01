# Pneumonia-Detection
This project aims to use Machine Learning algorithm to classify users based on the presence of Pneumonia using their chest X-rays

Pneumonia is a lung infection that ranges from mild to life-threatening consequences. In order to identify the infection, radiologists examine chest X-ray images to identify possible infection. By analyzing the "Chest X-Ray Images (Pneumonia)" dataset on Kaggle, we aim to use machine learning algorithms to help identify the presence of Pneumonia in patients and to facilitate diagnosis at a larger scale.

NOTE:

To download the dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia, you will need the "kaggle.json" file (Kaggle API token) to use the given extraction code in this repository

Steps to download the Kaggle API token.
1. Create an account on Kaggle/Sign In
2. Go to “Account”, go down the page, and find the “API” section.
3. Click the “Create New API Token” button.
4. The “kaggle.json” file will be downloaded.

You will require the following imports to use this code:

import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import cv2
import os
import pathlib
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.utils import resample

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score #weighted for imbalanced dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

from sklearn.model_selection import GridSearchCV

The following algorithms have been explored in this project:
1. Decision Tree Classifier
2. Random Forest Ensemble
3. K-Nearest Neighbors classifier
4. Convolutional Neural Networks
5. Transfer Learning

Additional Analysis done on the project includes:
1. Exploratory data analysis and visualizations
2. Principal Component Analysis
3. Class Imbalance Correction using upsampling and downsampling
4. Using Grid Search for finding suitable hyperparameters
