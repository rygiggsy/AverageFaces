#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file generates the facial landmarks of the footballers.
We will only select the most confident 'face' in each image. 

Created on Sat Oct  7 01:14:58 2017

@author: nganmeng.tan
"""

import numpy as np  
import cv2  
import dlib  
import os
import glob
from matplotlib import pyplot as plt
   
#image_path =  'test/' 
image_path =  'Top100MaleFootballer'


# Init frontal face detector
detector = dlib.get_frontal_face_detector()
predictor_path= 'predictor/shape_predictor_68_face_landmarks.dat' 
# create the landmark predictor  
predictor = dlib.shape_predictor(predictor_path)  

# Read all jpg images in folder.
for f in glob.glob(os.path.join(image_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = cv2.imread(f)

    #faces = detector(img, 1) 
    faces, scores, idx = detector.run(img, 1)

    print("Found {0} faces!".format(len(faces)))  
    if len(faces) == 0 : 
        # Break if no faces found! 
        print("{0} faces found!".format(len(faces)))  
        break        
    else:
        for i, d in enumerate(faces):
            # we only want the highest confidence scores 
            if scores[i] == max(scores):
                #print("Index {}, Detection {}, score: {}, face_type:{}".format(i, d, scores[i], idx[i]))
                x =  int(d.left()); y =  int(d.top()); x_w = int(d.right()); y_h = int(d.bottom());
                dlib_rect = dlib.rectangle(x, y, x_w, y_h) 
                #print (dlib_rect )
                detected_landmarks = predictor(img, dlib_rect).parts()  
               
                cv2.rectangle(img,(x,y),(x_w,y_h),(255,255,0),2)
                roi_color = img[y:y_h, x:x_w]
                landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])  
                filename = f[0:len(f)-3] + 'txt'     
              
                np.savetxt(filename, landmarks, fmt='%d',delimiter=' ')   # X is an array
                
                #========== Display Purposes: You can Disable safely. 
                # Display the landmark points
                # copying the image so we can see side-by-side  
                image_copy = img.copy()  
                
                for idx, point in enumerate(landmarks):  
                    pos = (point[0, 0], point[0, 1])  
                    # annotate the positions  
                    cv2.putText(image_copy, str(idx), pos,  
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,  
                       fontScale=0.4,  
                       color=(0, 0, 255))  
                    # draw points on the landmark positions  
                    cv2.circle(image_copy, pos, 3, color=(0, 255, 255))  
                    
                plt.axis("off")
                plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)) 
                plt.show()