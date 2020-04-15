import time

from imutils import contours
from skimage import measure
import imutils
import cv2
import matplotlib
import matplotlib.image
import numpy as np

def num_vortices(image, SHOW_IMG=True):
    """param image is numpy NxN matrix"""
    np.save("vortices_test",image)
    thresh = cv2.threshold(image, 1e-4, 1, cv2.THRESH_BINARY)[1]
    cv2.waitKey(0)
    cv2.imshow("Thresholded",thresh)
    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        mask = cv2.add(mask, labelMask)

    # find the contours in the mask
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
    hierarchy = hierarchy[0]  # grab actual hierarchy data

    # show the image and the vortices it found
    if SHOW_IMG:
        matplotlib.image.imsave("vortices.png", image, cmap='afmhot')
        pretty_image = cv2.imread("vortices.png")

    numvortices = 0
    for currentContour, currentHierarchy in zip(contours, hierarchy):
        ((cX, cY), radius) = cv2.minEnclosingCircle(currentContour)
        if currentHierarchy[2] < 0: # no child
            numvortices += 1
            # these are the innermost child components
            if SHOW_IMG:
                cv2.circle(pretty_image, (int(cX), int(cY)), int(radius+10), (0, 0, 255), 1)
        elif currentHierarchy[3] < 0: 
            # these are the outermost parent components
            cv2.circle(pretty_image, (int(cX), int(cY)), int(radius+10), (0, 255, 0), 2)
    
    if SHOW_IMG:
        # show the output image
        cv2.imshow("Image", pretty_image)
        cv2.waitKey(0)
        cv2.imwrite("vortices_detected.png", pretty_image) 
    return numvortices
    
if __name__ == "__main__":
        
    # Load the image
    img = np.load("vortices_test.npy")
    vortices =  num_vortices(img, SHOW_IMG=True)
    print("Number of vortices:",vortices)
    
