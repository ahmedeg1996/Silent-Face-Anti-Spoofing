# import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
# construct the argument parser and parse the arguments


def face_detection(image):
	
	# load dlib's HOG + Linear SVM face detector
	detector = dlib.get_frontal_face_detector()
	# load the input image from disk, resize it, and convert it from
	# BGR to RGB channel ordering (which is what dlib expects)
	image = cv2.imread(image)
	image = imutils.resize(image, width=600)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# perform face detection using dlib's face detector
	start = time.time()
	rects = detector(rgb, 1)
	end = time.time()
	return rects


