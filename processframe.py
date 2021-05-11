# import the necessary packages
from webcamvideostream import WebcamVideoStream
from kalman import kalman
from fps import FPS
import numpy.linalg as la
import argparse
import imutils
import math
import cv2 as cv
import numpy as np
import datetime
#from calcposition import *

def FindObject(mask,lower,upper,min_area):
	
	########## FIND OBJECTS ##########
	# Loop over the contours to find all objects
	contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)


	# Find biggest object using contourArea
	num = 0
	biggreen = None
	bigarea = 0
	for c in contours:
		area = cv.contourArea(c)
		if area==0:
			continue
		if area > bigarea:
			bigarea = area
			biggreen = c
			pos = num
		num += 1

	# Noise filter
	if biggreen is not None and bigarea > min_area:
		return biggreen
	else:
		return None

def BoundingBox(contour):
	# Calculate bounding box
	(x, y, w, h) = cv.boundingRect(contour)
	# Calculate x and y of the center point
	xo = x + int(w/2)
	yo = y + int(h/2)
	return (xo, yo, w, h)



def Coordinates3d(	x1, y1, x2, y2,
					res_x1,res_y1,res_x2,res_y2,
					aov_x1,aov_y1,aov_x2,aov_y2,
					c1_x,c1_y,c1_z,c2_x,c2_y,c2_z,
					average_x, average_y):
	def GetAngle(dim, resolution, aov):
		angleleft = aov * (dim/resolution)
		angle = aov - angleleft + (180-aov)/2
		return angle
	def GetFunc(angle, cx, cy):
		# Function: y = ax + b
		if angle != 90:
			a = math.tan(angle * (math.pi/180))
			b = cy - (a * cx)
			return (a,b)
		return (0,0)

	angle_x1 = GetAngle(x1, res_x1, aov_x1)
	angle_x2 = GetAngle(x2, res_x2, aov_x2)

	(a1,b1) = GetFunc(angle_x1, c1_x, c1_y)
	(a2,b2) = GetFunc(angle_x2, c2_x, c2_y)

	xo = None
	yo = None
	if a1-a2 != 0:
		xo = (b2-b1)/(a1-a2)
		yo = a1*xo + b1
	if (a1==0 and b1==0) or (a2==0 and b2==0):
		xo = average_x
		yo = average_y
	
	angle_z1 = GetAngle(y1, res_y1, aov_y1)
	angle_z2 = GetAngle(y2, res_y2, aov_y2)

	angle_z = (angle_z1+angle_z2)/2
	#print(angle_z)
	(a3,b3) = GetFunc(angle_z-90, c1_y, c1_z)

	zo = a3 * yo + b3
	return -xo,yo,zo



def Draw(frame,x,y,w,h):
	font = cv.FONT_HERSHEY_SIMPLEX
	cv.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 1)
	cv.circle(frame, center=(x, y), radius=int((w+h)/24), color=(0, 0, 255), thickness=-1, lineType=8, shift=0)
	text = "Coordinates:" + str(x) + ", " + str(y)
	cv.putText(frame, format(text), (10, 20), font, 0.5, (0, 0, 255), 2)
	return frame
