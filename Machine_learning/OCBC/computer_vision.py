import cv2
import numpy as np

img = cv2.imread(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\OCBC\boxes2.jpg", 0)
seed_pt = (25, 25)
fill_color = 0
mask = np.zeros_like(img)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
for th in range(30, 120):
    prev_mask = mask.copy()
    mask = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.floodFill(mask, None, seed_pt, fill_color)[1]
    mask = cv2.bitwise_or(mask, prev_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

n_centers = cv2.connectedComponents(mask)[0]
print('There are %d boxes in the image.'%n_centers)