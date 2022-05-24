# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import argparse
import utils
import cv2

# Create trackbars
utils.createTrackbar()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(1)


while True:
	# (blur_size, k_size, mor_iters, min_area, lowThres, highThres, brightness, contrast, type_edge)
	params = utils.getValuesFromTrackBars()

	# Read image
	_, image = cap.read()

	# # Resize image
	# image = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3))


	# Change brightness and contrast
	effect = utils.controller(image, params[6], params[7])

	# Convert into grayscale and blur
	gray = cv2.cvtColor(effect, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (params[0], params[0]), 0) 
                                            

	# Perform Edge Detection
	if params[8] == 0:
		edge = cv2.Canny(blur, params[4],  params[5])
		name_edge = 'Canny'
	elif params[8] == 1:
		edge = utils.sobel_edge_detection(blur, params[4], params[1])
		name_edge = 'Sobel'
	elif params[8] == 2:
		edge = utils.scharr_edge_detection(blur, params[4])
		name_edge = 'Scharr'
	else:
		edge = utils.laplacian_edge_detection(blur, params[4], params[1])
		name_edge = 'Laplacian'
	edge = edge.astype('uint8')


	# # Try different morphology kernel
	# edged = cv2.dilate(edge, (mor_size, mor_size), iterations=mor_iters)
	# edged_morphology = cv2.erode(edged, (mor_size, mor_size), iterations=mor_iters)

	# Perform morphology 
	edged = cv2.dilate(edge,None, iterations=params[2])
	edged_morphology = cv2.erode(edged, None, iterations=params[2])

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
		flag = True
		pixelsPerMetric = 0
		for c in cnts:
			# Ignore contours if it not large enough
			if cv2.contourArea(c) < params[3]:
				continue
			# compute bounding box
			box = cv2.minAreaRect(c)
			box = cv2.boxPoints(box)
			box = np.array(box, dtype="int")

			# order the points in the contour such that they appear
			# in top-left, top-right, bottom-right, and bottom-left order
			box = utils.order_points(box)

			# # Drawing bounding box
			# x, y, w, h = cv2.boundingRect(c)
			# cv2.rectangle(res,(x,y),(x+w,y+h),(255,0,0),2)
			
			# Drawing contours
			cv2.drawContours(res, [box.astype("int")], -1, (0, 255, 0), 2)	

			# compute the midpoint
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
			if flag:
				while pixelsPerMetric == 0:
					pixelsPerMetric = dB / args["width"]
				flag = False

			# compute the size of the object
			dimA = dA / pixelsPerMetric
			dimB = dB / pixelsPerMetric

			# draw the object sizes on the image
			cv2.putText(res, "{:.1f}mm".format(dimB),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(res, "{:.1f}mm".format(dimA),
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
			# and then convert the distance in pixels to distance in units
			D = dist.euclidean((xA, yA), (xB, yB)) / pixelsPerMetric
			(mX, mY) = utils.midpoint((xA, yA), (xB, yB))
			cv2.putText(res, "{:.1f}mm".format(D), (int(mX), int(mY - 10)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

			refCoords = objCoords

		# Stacking images
		imageArray = ([image, effect, gray], [edge, img_cnts, res])
		labels = [["Original","Effect","Gray",],
				[name_edge,"Contours", "Result"]]
		stackedImage = utils.stackImages(imageArray,0.6, labels)	
	else:
		# Stacking images
		imageArray = ([image, effect, gray], [edge, image, image])
		labels = [["Original","Effect","Gray",],
				[name_edge,"Contours", "Result"]]
		stackedImage = utils.stackImages(imageArray,0.6, labels)	
	# Show result
	cv2.imshow("WINDOW",stackedImage)
	if cv2.waitKey(1) == ord('q'):
		break
