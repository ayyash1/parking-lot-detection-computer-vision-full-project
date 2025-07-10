import pickle
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize

from utils import get_parking_slot_bboxes, slot_availability, calc_diff

#paths
video_path = 'ParkingLotDetection/Data/parking_1920_1080_loop.mp4'
mask_path = 'ParkingLotDetection/Data/mask_1920_1080.png'

cap = cv2.VideoCapture(video_path)

mask = imread(mask_path) 

#convert into a grayscale image
if len(mask.shape) == 3:
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

slots = get_parking_slot_bboxes(connected_components)

slot_status_all = [None for j in slots]
diffs = [None for j in slots]

previous_frame = None

step = 30 #how often should we classify the parking lots
frame_num = 0

while True:

    ret, frame = cap.read()

    if frame_num % step ==  0 and previous_frame is not None:
        for slot_idx, slot in enumerate(slots):
            x1, y1, w, h = slot

            #crop the slots in the video
            slot_crop = frame[y1:y1+h, x1:x1+w, :]

            diffs[slot_idx] = calc_diff(slot_crop,previous_frame[y1:y1+h, x1:x1+w, :])

        #print([diffs[j] for j in np.argsort(diffs)][::-1])
        #plot a histogram to detect outliers
        # plt.figure()
        #plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1])
        #if frame_num==300:
            #plt.show()

    #updating the parking slot status
    if frame_num % step ==  0:

        if previous_frame is None:
            arr_ = range(len(slots))
        else:
            #from histogram, we detect outliers and set threshold as 0.4 to remove outliers
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs)> 0.4]
        
        for slot_idx in arr_:

            slot = slots[slot_idx]
            x1, y1, w, h = slot

            #crop the slots in the video
            slot_crop = frame[y1:y1+h, x1:x1+w, :]
            slot_status = slot_availability(slot_crop)

            slot_status_all[slot_idx] = slot_status

    if frame_num % step ==  0:
        previous_frame = frame.copy()

    #draw bounding boxes for parking lots
    for slot_idx, slot in enumerate(slots):

        slot_status = slot_status_all[slot_idx]
        x1, y1, w, h = slots[slot_idx]

        if slot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255,0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
        
    #add a counter
    cv2.rectangle(frame, (80,20), (550, 80), (0,0,0), -1)
    cv2.putText(frame, "Available slots: {} / {}".format(str(sum(slot_status_all)), str(len(slot_status_all))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)


    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_num +=1


cap.release()
cv2.destroyAllWindows()