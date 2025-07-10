import cv2
import numpy as np
import pickle

from skimage.io import imread
from skimage.transform import resize

EMPTY = True
NOT_EMPTY = False

#load classifer model
clf = pickle.load(open('ParkingLotDetection/TrainClassifier/model.pkl', "rb"))

def slot_availability(slot_bgr):
    
    flat_data = []

    img_resized = resize(slot_bgr, (15,15,3 ))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_pred = clf.predict(flat_data)

    if y_pred == 0:
        return EMPTY
    else:
        return NOT_EMPTY

def get_parking_slot_bboxes(connected_components):

    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1

    for i in range(1, totalLabels):

        #extract the coordinates
        x1 = int(values[i, cv2.CC_STAT_LEFT]* coef)
        y1 = int(values[i, cv2.CC_STAT_TOP]* coef)
        w = int(values[i, cv2.CC_STAT_WIDTH]* coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT]* coef)

        slots.append([x1, y1, w, h])

    
    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))