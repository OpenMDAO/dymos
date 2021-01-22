import numpy as np

class Track:
	def __init__(self,segments):
		self.segments = segments
		self.cornerspeeds = np.zeros(len(segments),dtype=np.complex)

	def getSegmentType(self,num):
		return self.segments[num][0]

	def getSegmentLength(self,num):
		return self.segments[num][1]

	def getSegmentRadius(self,num):
		return self.segments[num][2]

	def getCornerDirection(self,num):
		return self.segments[num][3]

	def setCornerSpeed(self,num,speed):
		self.cornerspeeds[num] = speed

	def getCornerSpeed(self,num):
		return self.cornerspeeds[num]

	def getTotalLength(self):
		length = 0
		for i in range(len(self.segments)):
			if self.segments[i][0] == 0:
				#straight
				length += self.getSegmentLength(i)
			else:
				#corner
				length += self.getSegmentLength(i)*self.getSegmentRadius(i)
		return length