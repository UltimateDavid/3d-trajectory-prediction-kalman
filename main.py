# USAGE
# python main.py --display 1
# python main.py --video videos/example_04.mp4


# import the necessary packages
from webcamvideostream import WebcamVideoStream
from kalman import kalman
from fps import FPS
from processframe import FindObject,BoundingBox,Draw,Coordinates3d
import numpy.linalg as la
import argparse
import imutils
import math
import cv2 as cv
import numpy as np
import datetime

font = cv.FONT_HERSHEY_SIMPLEX

# Minimum area of an object to be recognized (to avoid noise)
min_area = 20

# Range of green color in HSV
lower = np.array([40,100,30])
upper = np.array([100,255,255])

video_file1 = 'videos/example_05_c2.avi'
video_file2 = 'videos/example_05.avi'
# video_file1 = None
# video_file2 = None
pause = False 



################ Camera initialisation ################
# Resolution
res_x1 = 640
res_y1 = 480
res_x2 = 640
res_y2 = 480

# Angle of View
aov_x1 = 67
aov_y1 = 48
aov_x2 = 67
aov_y2 = 48

# Position in 3D
c1_x = 0
c1_y = 0
c1_z = 0
c2_x = c1_x + 4
c2_y = 0
c2_z = 0

frame_w = 600
frame_h = int(frame_w * (res_y1/res_x1))



################ Kalman initialisation ################
# Starting Values of the Kalman Filter

a = np.array([0, 0, -980]) # Which variable is being calculated? X Y or Z
 
# State Matrix: x,y,vx,vy
mu = np.array([0,0,0,0,0,0])
mu3 = np.array([0,0,0,0,0,0])
# Covariance matrix: uncertainty
P = np.diag([1000,1000,1000,1000,1000,1000])**2
P3 = np.diag([1000,1000,1000,1000,1000,1000])**2

speed = 0.25
fps = 42
dt = 1/(fps/speed) # every timestep
noise = 4

sigmaM = 0.001 # model noise 
sigmaZ = 3*noise # average noise of the imaging process

Q = sigmaM**2 * np.eye(6) # Noise of F
R = sigmaZ**2 * np.eye(3) # Noise of H

F = np.array( 
		[1, 0, 0, dt, 0, 0,
		 0, 1, 0, 0, dt, 0,
		 0, 0, 1, 0, 0, dt,
		 0, 0, 0, 1, 0, 0,
		 0, 0, 0, 0, 1, 0,
		 0, 0, 0, 0, 0, 1]).reshape(6,6)

B = np.array( 
		[dt**2/2, 0, 0,
		 0, dt**2/2, 0,
		 0, 0, dt**2/2,
		 dt, 0, 0, 
		 0, dt, 0,
		 0, 0, dt]).reshape(6,3)

B = np.array( 
		[dt**2/2, 0, 0,
		 0, dt**2/2, 0,
		 0, 0, dt**2/2,
		 dt, 0, 0, 
		 0, dt, 0,
		 0, 0, dt]).reshape(6,3)

H = np.array(
	[1,0,0,0,0,0,
	 0,1,0,0,0,0,
	 0,0,1,0,0,0]).reshape(3,6)

# res = [(mu,P)] # will be all the matrices in the Kalman Filter
#    mu is position,velocity en P is uncertainty
res=[] 



# Storage of measurements
list3dX = []
list3dY = []
list3dZ = []

list2dX_1 = []
list2dY_1 = []
list2dX_2 = []
list2dY_2 = []

list3dXe = []
list3dYe = []
list3dZe = []

lastframe1 = None
lastframe2 = None

xe = None

average_x = 0
average_y = 0


################ START VIDEOSTREAM ################
# Check if we have a video or a webcam
if (video_file1 is not None) and (video_file2 is not None):
	print("[INFO] setting up videofile...")
	stream1 = cv.VideoCapture(video_file1)
	stream2 = cv.VideoCapture(video_file2)
	fps1 = stream1.get(cv.CAP_PROP_FPS)
	fps2 = stream2.get(cv.CAP_PROP_FPS)
# otherwise, we are reading from webcams
else:
	print("[INFO] setting up webcams...")
	stream1 = WebcamVideoStream(src=0).start()
	stream2 = WebcamVideoStream(src=0).start()
# Start framecount
streamfps = FPS().start()
streamfps = FPS().start()

out = cv.VideoWriter('outframe1.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15, (frame_w, frame_h))
out2 = cv.VideoWriter('outframe2.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15, (frame_w, frame_h))
out3 = cv.VideoWriter('outgraph.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15, (frame_w, frame_h))

######################## START CAPTURING ########################
# loop over every frame
print("[INFO] starting the stream...")
while(True):
	key = cv.waitKey(40) & 0xFF
	if key == ord("p"): P = np.diag([100,100,100,100,100,100])**2 # Make the filter less uncertain
	if key == ord("q") or key == 27: break # quitting when ESCAPE or q is pressed
	if key == ord(" "): pause =not pause # pause when spacebar is pressed, unpause when pressed again
	if(pause): continue



	################ GRAB FRAME AND PROCESS ################
	# grab the frame from the stream and resize it
	(grabbed1, frame_raw1) = stream1.read()
	(grabbed2, frame_raw2) = stream2.read()
	# Check if the frames have been grabbed
	if grabbed1 is False or grabbed2 is False:
		# Video has ended, but we still want to watch the end result
		continue
	# Check if the frames are not None or empty
	if not(isinstance(frame_raw1, np.ndarray)) or not(isinstance(frame_raw2, np.ndarray)):
		print("Oops, something went wrong with the frames")
		break
	# Check if current and last frames are the same
	if frame_raw1 is lastframe1 or frame_raw2 is lastframe2:
		break
	frame1 = imutils.resize(frame_raw1, width=frame_w, height=frame_h)
	frame2 = imutils.resize(frame_raw2, width=frame_w, height=frame_h)
	graph3d = np.zeros((frame_h,frame_w,3), np.uint8)

	################ FIND GREEN PIXELS ################
	# Convert BGR to HSV
	hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
    # Threshold the HSV image to get only green colors
	mask1 = cv.inRange(hsv1, lower, upper)
	mask2 = cv.inRange(hsv2, lower, upper)



	################ Find object ################
	# get contours, find biggest contour (which is the ball), noise filter
	contour1_ball = FindObject(mask1, lower,upper, min_area)
	contour2_ball = FindObject(mask2, lower,upper, min_area)

	if (contour1_ball is not None or contour2_ball is not None) and streamfps.frames() > 45: # Check if there is any green object 
		(x1, y1, w1, h1) = BoundingBox(contour1_ball)
		(x2, y2, w2, h2) = BoundingBox(contour2_ball)
		list2dX_1.append(x1)
		list2dY_1.append(y1)
		list2dX_2.append(x2)
		list2dY_2.append(y2)

		if len(list3dYe)!=0:
			average_x = sum(list3dXe) / len(list3dXe)
			average_y = sum(list3dYe) / len(list3dYe)

		xo,yo,zo = Coordinates3d(x1,  y1,  x2,  y2, # Position of ball camera 1 and 2
					res_x1, res_y1, res_x2, res_y2, # Resolution of screen camera 1 and 2		
					aov_x1, aov_y1, aov_x2, aov_y2, # Angle of View camera 1 and 2
					c1_x,c1_y,c1_z, c2_x,c2_y,c2_z,	# Camera Position camera 1 and 2
					average_x, average_y)			# Average Y in the case there is a problem
		
		# 3D-Coordinates:
		# X-axis: 	Horizontal (Parallel to the camera's), 
		#			determined by the x-values (and a little bit the difference between them)
		# 			LINEAIR
		# Y-axis:	Depth (how far away is the ball)
		#			determined by the difference in x-values of the camera's
		# 			LINEAIR
		# Z-axis:	Vertical Height (How high is the ball above the ground), 
		#			determined by the y-values
		# 			PARABOLEA

		if xo is not None:
			list3dX.append(xo)
			list3dY.append(yo)
			list3dZ.append(zo)
		print(xo,yo,zo)



		######## KALMAN FILTER Position #########
		mu,P,pred = kalman(mu,P,F,Q,B,a,np.array([xo,yo,zo]),H,R) 
		
		# Storing of matrices
		res = [(mu,P)]

		xe = [mu[0] for mu,_ in res] # Estimated position
		ye = [mu[1] for mu,_ in res]
		ze = [mu[2] for mu,_ in res] 
		xu = [2*np.sqrt(P[0,0]) for _,P in res] # uncertainty of estimated position
		yu = [2*np.sqrt(P[1,1]) for _,P in res] # uncertainty of estimated position
		zu = [2*np.sqrt(P[1,1]) for _,P in res] # uncertainty of estimated position
		# Storing of estimations
		list3dXe.append(xe[0])
		list3dYe.append(ye[0])
		list3dZe.append(ze[0])



		######## KALMAN FILTER Prediction #########
		mu2 = mu 
		P2 = P 
		res2 = []

		for _ in range(int(fps*3)): 
			mu2,P2,pred2 = kalman(mu2,P2,F,Q,B,a,None,H,R) 
			res2 += [(mu2,P2)]

		xp = [mu2[0] for mu2,_ in res2] 
		yp = [mu2[1] for mu2,_ in res2] 
		zp = [mu2[2] for mu2,_ in res2] 
		xpu= [2*np.sqrt(P[0,0]) for _,P in res2] 
		ypu= [2*np.sqrt(P[1,1]) for _,P in res2]
		zpu= [2*np.sqrt(P[2,2]) for _,P in res2]




		# Predict next few x on the frames
		mu3,P3,pred3 = kalman(mu3,P3,F,Q,B,a,np.array([list2dX_1[-1],0,0]),H,R) 
		mu4,P4 = mu3,P3
		res4 = []
		for _ in range(int(fps*3)): 
			mu4,P4,pred4 = kalman(mu4,P4,F,Q,B,a,None,H,R) 
			res4 += [(mu4,P4)]
		xframe = [mu4[0] for mu4,_ in res4]
		xframeu = [2*np.sqrt(P[0,0]) for _,P in res4]



	################ DRAW ################
	if list2dX_1:
		frame1 = Draw(frame1,x1,y1,w1,h1)
		frame2 = Draw(frame2,x2,y2,w2,h2)
		
		# horizontal axis
		cv.line(graph3d, (0,int(frame_w/2)+15), (frame_w,int(frame_w/2)+15), (220, 220, 220), thickness=1, lineType=8, shift=0)
		# vertical axis
		#cv.line(graph3d, (int(frame_h/2)+60,0), (int(frame_h/2)+60,frame_h), (220, 220, 220), thickness=1, lineType=8, shift=0)
		

		# Draw every measurement so far on the frames
		for n in range(len(list2dX_1)): 
			cv.circle(frame1,(int(list2dX_1[n]),int(list2dY_1[n])),3, (0, 255, 0),-1)
		for n in range(len(list2dX_2)): 
			cv.circle(frame2,(int(list2dX_2[n]),int(list2dY_2[n])),3, (0, 255, 0),-1)
	
		# Make visual representation of 3d position on a new frame
		for n in range(len(list3dX)): 
			cv.circle(graph3d,(list2dX_1[n]-40, int(3* list3dX[n]+frame_h/2+90)),3, (0, 220, 0),-1)
			cv.circle(graph3d,(list2dX_1[n]-40, int(3*-list3dY[n]+frame_h/2+90)),3, (220, 50, 50),-1)
			cv.circle(graph3d,(list2dX_1[n]-40,-int(3* list3dZ[n]-frame_h/2-90)),3, (00, 0, 220),-1)


		# Kalman filter position
		for n in range(len(list3dXe)): 
			cv.circle(graph3d,(list2dX_1[n]-40, int(3* list3dXe[n]+frame_h/2+90)),3, (0, 150, 0),-1)
			cv.circle(graph3d,(list2dX_1[n]-40, int(3*-list3dYe[n]+frame_h/2+90)),3, (180, 20, 20),-1)
			cv.circle(graph3d,(list2dX_1[n]-40,-int(3* list3dZe[n]-frame_h/2-90)),3, (00, 0, 150),-1)

		# Draw the predicted paths
		for n in range(len(xp)): 
			uncertaintyP = (xpu[n]+ypu[n]+zpu[n])/3 
			# Draw prediction (circles), with uncertainty as radius
			cv.circle(graph3d,(int(xframe[n])-45, int(3* xp[n]+frame_h/2+90)),int(uncertaintyP), (0, 100, 0))
			cv.circle(graph3d,(int(xframe[n])-45, int(3* xp[n]+frame_h/2+90)),3, (0, 255, 0),-1)
			
			cv.circle(graph3d,(int(xframe[n])-45, int(3*-yp[n]+frame_h/2+90)),int(uncertaintyP), (100, 0, 0))
			cv.circle(graph3d,(int(xframe[n])-45, int(3*-yp[n]+frame_h/2+90)),3, (255, 0, 0),-1)
			
			cv.circle(graph3d,(int(xframe[n])-45,-int(3* zp[n]-frame_h/2-90)),int(uncertaintyP), (0, 0, 100))
			cv.circle(graph3d,(int(xframe[n])-45,-int(3* zp[n]-frame_h/2-90)),3, (0, 0, 255),-1)
	

	cv.putText(graph3d, format("X"), (10, 20), font, 0.5, (0, 255, 0), 2)
	cv.putText(graph3d, format("Y"), (30, 20), font, 0.5, (255, 0, 0), 2)
	cv.putText(graph3d, format("Z"), (50, 20), font, 0.5, (0, 0, 255), 2)

	cv.imshow("Frame 1", frame1)
	#cv.imshow("Vision", mask1)
	cv.imshow("Frame 2", frame2)
	#cv.imshow("Vision", mask2)
	cv.imshow("Position", graph3d)
	out.write(frame1)
	out2.write(frame2)
	out3.write(graph3d)
	if not(grabbed1 is False or grabbed2 is False):
		# update the FPS counter
		streamfps.update()
		#fps = streamfps.fps_now()
	
 	
# stop the timer and display FPS information
streamfps.stop()
print("[INFO] elasped time: {:.2f}".format(streamfps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps))

# do a bit of cleanup
stream1.release()
stream2.release()
out.release()
out2.release()
out3.release()
cv.destroyAllWindows()
