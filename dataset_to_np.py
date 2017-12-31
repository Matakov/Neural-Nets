import numpy as np 
import os
import cv2

imgs = []
for i, file in enumerate(sorted(os.listdir())):
    if i%100 == 0:
        print (i)
    if file == "ssims.txt":
        f = open("ssims.txt", "r")
        ssims = f.readlines()
        np.save("Y_data", np.asarray(ssims).astype(np.float64))
        continue
    if file == "dataset_to_np.py":
        continue
    imgs.append(cv2.imread(file,0))
np.save("X_data", np.asarray(imgs))



    
