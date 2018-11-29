# import packages
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import pandas as pd

# fixed-sizes for image in this case
# if images are in different size
# fixed_size = tuple((120, 80))

# path to training data
train_path = "data/ISIC-2017_Training_Data"

# no of trees for Random Forests
num_trees = 100

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# empty lists to hold feature vectors
train_features = []

#####################################
# extract features for X_train
#####################################
# loop over the training data
dir = os.path.join(train_path, "")
for file in glob.glob(os.path.join(dir, '*.jpg')):

    # read the image and resize it to a fixed-size
    image = cv2.imread(file)
    # image = cv2.resize(image, fixed_size)

    # Global Feature extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    # Concatenate global features
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    train_features.append(global_feature)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(train_features)

# save the feature vector using HDF5
x_train = h5py.File('Extracted_Features/x_train.h5', 'w')
x_train.create_dataset('dataset_1', data=np.array(rescaled_features))

# save the the label vector using HDF5
df = pd.read_csv("data/ISIC-2017_Training_Part3_GroundTruth.csv")
labels = df.iloc[:, 1:2]
y_train = h5py.File('Extracted_Features/y_train.h5', 'w')
y_train.create_dataset('dataset_1', data=np.array(labels))

x_train.close()
y_train.close()


#####################################
# extract features for X_test
#####################################

# empty lists to hold feature vectors and labels
validation_features = []

validation_path = "data/ISIC-2017_Validation_Data"
dir = os.path.join(validation_path, "")
for file in glob.glob(os.path.join(dir, '*.jpg')):
    # read the image and resize it to a fixed-size
    image = cv2.imread(file)
    # image = cv2.resize(image, fixed_size)

    # Global Feature extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    # Concatenate global features
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    validation_features.append(global_feature)

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(validation_features)

# save the feature vector using HDF5
x_test = h5py.File('Extracted_Features/x_test.h5', 'w')
x_test.create_dataset('dataset_1', data=np.array(rescaled_features))

# save the the label vector using HDF5
df = pd.read_csv("data/ISIC-2017_Validation_Part3_GroundTruth.csv")
labels = df.iloc[:, 1:2]
y_test = h5py.File('Extracted_Features/y_test.h5', 'w')
y_test.create_dataset('dataset_1', data=np.array(labels))

x_test.close()
y_test.close()
