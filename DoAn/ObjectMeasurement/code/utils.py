import cv2
import numpy as np
from scipy.spatial import distance as dist
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
 
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to bottom
    # return (x, y, w, h)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes
def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
    
def sobel_edge_detection(img, thresh, k):
    img_Sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    abs_img_Sobelx = cv2.convertScaleAbs(img_Sobelx)

    img_Sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
    abs_img_Sobely = cv2.convertScaleAbs(img_Sobely)

    img_Sobel = cv2.addWeighted(abs_img_Sobelx, 0.5, abs_img_Sobely, 0.5, 0)

    ret, threshold_img = cv2.threshold(img_Sobel, thresh, 255, cv2.THRESH_BINARY)

    return threshold_img

def scharr_edge_detection(img, thresh):

    
    img_Scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
    abs_img_Scharrx = cv2.convertScaleAbs(img_Scharrx)

    img_Scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
    abs_img_Scharry = cv2.convertScaleAbs(img_Scharry)

    img_Scharr = cv2.addWeighted(abs_img_Scharrx, 0.5, abs_img_Scharry, 0.5, 0)

    ret, threshold_img = cv2.threshold(img_Scharr, thresh, 255, cv2.THRESH_BINARY)

    return threshold_img


def laplacian_edge_detection(img, thresh, k):
    import matplotlib.pyplot as plt
    
    img_Laplacian = cv2.Laplacian(img,cv2.CV_64F, ksize=k)
    img_Laplacian = cv2.convertScaleAbs(img_Laplacian)

    ret, threshold_img = cv2.threshold(img_Laplacian, thresh, 255, cv2.THRESH_BINARY)

    return threshold_img

def controller(img, brightness=255, contrast=127):

    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
 
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max = 255
        else:
            shadow = 0
            max = 255 + brightness
 
        al_pha = (max - shadow) / 255
        ga_mma = shadow
 
        # The function addWeighted
        # calculates the weighted sum
        # of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)
 
    else:
        cal = img
 
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
 
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    return cal

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def createTrackbar():
    cv2.namedWindow('TRACK BAR', cv2.WINDOW_AUTOSIZE)

    def null(x):
        pass

    cv2.createTrackbar("Blur", "TRACK BAR", 1, 15, null)
    cv2.createTrackbar("MorIters", "TRACK BAR", 1, 15, null)
    cv2.createTrackbar("Min area", "TRACK BAR", 0, 5000, null)
    cv2.createTrackbar("Low Threshold", "TRACK BAR", 0, 255, null)
    cv2.createTrackbar("High Threshold", "TRACK BAR", 0, 255, null)
    cv2.createTrackbar("Brightness", "TRACK BAR", 255, 255 * 2, null)
    cv2.createTrackbar("Contrast", "TRACK BAR", 127, 127 * 2, null)
    cv2.createTrackbar("Edge", "TRACK BAR", 0, 3, null)
    cv2.createTrackbar("Kernel", "TRACK BAR", 1, 13, null)

def getValuesFromTrackBars():
    # Get values from trackbars
    blur_size = cv2.getTrackbarPos("Blur", "TRACK BAR")
    if blur_size % 2 == 0:
        blur_size += 1 
    k_size = cv2.getTrackbarPos("Kernel", "TRACK BAR")
    if k_size % 2 == 0:
        k_size += 1
    mor_iters = cv2.getTrackbarPos("MorIters", "TRACK BAR")
    min_area = cv2.getTrackbarPos("Min area", "TRACK BAR")
    lowThres = cv2.getTrackbarPos("Low Threshold", "TRACK BAR")
    highThres = cv2.getTrackbarPos("High Threshold", "TRACK BAR")
    brightness = cv2.getTrackbarPos("Brightness", "TRACK BAR")
    contrast = cv2.getTrackbarPos("Contrast", "TRACK BAR")
    type_edge = cv2.getTrackbarPos("Edge", "TRACK BAR")

    return (blur_size, k_size, mor_iters, min_area, lowThres, highThres, brightness, contrast, type_edge)