import time

from imutils import contours
from skimage import measure
import imutils
import cv2
import matplotlib
import numpy as np

def num_vortices(image):
    """param image is numpy NxN matrix"""
    np.save("vortices_test",image)
    # blurred = gaussian_filter(image, sigma=1)
    thresh = cv2.threshold(image, 1e-4, 1, cv2.THRESH_BINARY_INV)[1]
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

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_LIST  )
    print(cnts)
    cnts = imutils.grab_contours(cnts)
    print(cnts)
    if cnts:
        cnts = contours.sort_contours(cnts)[0]
    print(f"Found {len(cnts) - 2} vortices")
    # show the image and the vortices it found
    matplotlib.image.imsave("vortices.png", image, cmap='afmhot')
    matplotlib.image.imsave("vortices_thresh.png", thresh)

    pretty_image = cv2.imread("vortices.png")
    time.sleep(0.2)

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(pretty_image, (int(cX), int(cY)), int(radius+10),
            (0, 0, 255), 1)
        print(radius)
        # cv2.putText(pretty_image, "#{}".format(i + 1), (x, y - 15),
        # 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Image", pretty_image)
    cv2.waitKey(0)
    return len(cnts) - 2
