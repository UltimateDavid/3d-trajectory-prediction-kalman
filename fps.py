# import the necessary packages
import datetime
 
class FPS:
	def __init__(self):
		# Store the start time, end time, and total number of frames
		self._start = None
		self._end = None
		self._numFrames = 0
 
	def start(self):
		# Start the timer
		self._start = datetime.datetime.now()
		return self
 
	def stop(self):
		# Stop the timer
		self._end = datetime.datetime.now()
 
	def update(self):
		# Increment the total number of frames
		self._numFrames += 1
	
	def frames(self):
		# Return the total number of frames
		return self._numFrames

	def elapsed(self):
		# Return the total number of seconds
		return (self._end - self._start).total_seconds()
	
	def fps_now(self):
		return self._numFrames /((datetime.datetime.now() - self._start).total_seconds())

	def fps_tot(self):
		# Compute the (approximate) frames per second
		return self._numFrames / self.elapsed()