# import the necessary packages
import numpy as np
import glob
import cv2
import mahotas
import os
import csv
import pickle


bins = 8

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


# load the model
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

# path to test data
test_path = "data/ISIC-2017_Test_v2_Data"

with open('file.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['Name', 'Classe'])

    # loop through the test images
    for file in glob.glob(os.path.join(test_path, '*.jpg')):
        # read the image
        image = cv2.imread(file)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = loaded_model.predict(global_feature.reshape(1, -1))[0]

        # extract image name from the path & delete jpg extension
        image_name = os.path.basename(file)
        image_name = os.path.splitext(image_name)[0]
        if(prediction == 0):
            filewriter.writerow([image_name, 0])
        else:
            filewriter.writerow([image_name, 1])

