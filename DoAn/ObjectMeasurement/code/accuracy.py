# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import argparse
import utils
import cv2



# Xu1 - [23.5, 23.5]
# Xu1 - [23.5, 23.5]
# Card - [90, 53]
# Sac - [90, 42]
# Co - [31.5, 31.5]
# 2 hop - [90, 65]
# Sach - [145, 98]

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in mm)")
args = vars(ap.parse_args())

pixelsPerMetric = None
actual = [[23.5, 23.5], [90, 53], [31.5, 31.5], [90, 42], [90, 65], [90, 65]] # org2
# actual = [[23.5, 23.5],[31.5, 31.5], [90, 65], [90, 65],[23.5, 23.5],[145, 98]] # org
# actual = [[23.5, 23.5], [90, 65], [90, 65], [53, 90], [31.5, 31.5]]
predicted = []





image = cv2.imread(args["image"])

effect = utils.controller(image, 255, 127)


gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0) 			


# edge = cv2.Canny(blur, 100, 200)
# edge = utils.sobel_edge_detection(blur, 80, 3)
# edge = utils.scharr_edge_detection(blur, 90)
edge = utils.laplacian_edge_detection(blur, 15, 3)


edged = cv2.dilate(edge, None, iterations=1)
edged_morphology = cv2.erode(edged, None, iterations=1)

# find contours in image
contours, hierarchy = cv2.findContours(edged_morphology.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
if len(contours) != 0:
	# sort the contours from left-to-right
	(cnts, _) = utils.sort_contours(contours)

	# Drawing raw contours
	img_cnts = image.copy()
	cv2.drawContours(img_cnts, cnts, -1, (0, 0, 255), 3)

	res = image.copy()

	refCoords = None
	# loop over the contours individually
	for c in cnts:
		# Ignore contours if it not large enough
		if cv2.contourArea(c) < 500:
			continue
		# compute bounding box
		box = cv2.minAreaRect(c)
		box = cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bbox
		box = utils.order_points(box)

		cv2.drawContours(res, [box.astype("int")], -1, (0, 255, 0), 2)	
		# res = cv2.rectangle(image, (x, y),(x+w, y+h),(0,255,0),2)
		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = utils.midpoint(tl, tr)
		(blbrX, blbrY) = utils.midpoint(bl, br)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = utils.midpoint(tl, bl)
		(trbrX, trbrY) = utils.midpoint(tr, br)
	
		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if pixelsPerMetric is None:
			pixelsPerMetric = dB / args["width"]
			print(dB, pixelsPerMetric)

		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
		if dimA >= dimB:
			temp = [round(dimA, 2), round(dimB, 2)]
		else:
			temp = [round(dimB, 2), round(dimA, 2)]
		predicted.append(temp)
		# draw the object sizes on the image
		cv2.putText(res, "{:.2f}mm".format(dimB),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(res, "{:.2f}mm".format(dimA),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)


		# compute the center of the bounding box
		cX = np.average(box[:, 0])
		cY = np.average(box[:, 1])

		
		# if this is the first contour we are examining (i.e.,
		# the left-most contour), we presume this is the
		# reference object
		if refCoords is None:
			(tl, tr, br, bl) = box
			(tlblX, tlblY) = utils.midpoint(tl, bl)
			(trbrX, trbrY) = utils.midpoint(tr, br)

			refCoords = np.vstack([box, (cX, cY)])
			continue

		objCoords = np.vstack([box, (cX, cY)])

		xA, yA = refCoords[-1][0], refCoords[-1][1]
		xB, yB = objCoords[-1][0], objCoords[-1][1]
		color = (0, 0, 255)

		# draw circles corresponding to the current points and
		# connect them with a line
		cv2.circle(res, (int(xA), int(yA)), 5, color, -1)
		cv2.circle(res, (int(xB), int(yB)), 5, color, -1)
		cv2.line(res, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
		# compute the Euclidean distance between the coordinates,
		# and then convert the distance in pixels to distance in
		# units
		D = dist.euclidean((xA, yA), (xB, yB)) / pixelsPerMetric
		(mX, mY) = utils.midpoint((xA, yA), (xB, yB))
		cv2.putText(res, "{:.2f}mm".format(D), (int(mX), int(mY - 10)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

		refCoords = objCoords


cv2.imshow("RESULT",res)
cv2.imshow("EDGE",edge)
# cv2.imwrite(r"picture\result\SOBEL_FAIL_EVALUATION.png",res)
cv2.waitKey(0)

accu = []
for i, j in zip(actual, predicted):
	c1 = (abs(i[0] - j[0]) / i[0]) * 100
	c2 = (abs(i[1] - j[1]) / i[0]) * 100
	temp = 100 -  ((c1 + c2) / 2)
	accu.append(round(temp, 2))
print(accu)
print(predicted)
print(np.mean(np.array(accu)))