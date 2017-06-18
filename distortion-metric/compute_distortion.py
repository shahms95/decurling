#!/usr/bin/python
import cv2, sys
import numpy as np

def match_color(p1,p2,thres=5):
	return (abs(p1[0]-p2[0]) <=thres and abs(p1[1]-p2[1]) <=thres and abs(p1[2]-p2[2]) <=thres ) 


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
image = cv2.imread(sys.argv[1])
imageReconstructed = cv2.imread(sys.argv[2])

# colours = [[(0,0,205)]] 	#red
# colours = [[(0,205,0)]]	#green
# colours = [[(205,0,0)]]		#blue
s = 5

colours1 = [[(i*25,0,j*25) for i in range(1,10) ] for j in range(1,10)]		#for original image
colours2 = [[(45, 0, 45), (45, 0,75), (45, 0,102), (45, 0, 127), (45, 0, 151), (45, 0,173), (45, 0,195), (45, 0,215), (45, 0, 235)], [(75, 0, 45), (75, 0,75), (75, 0,102), (75, 0, 127), (75, 0, 151), (75, 0,173), (75, 0,195), (75, 0,215), (75, 0, 235)], [(102, 0, 45), (102, 0,75), (102, 0,102), (102, 0, 127), (102, 0, 151), (102, 0,173), (102, 0,195), (102, 0,215), (102, 0, 235)], [(127, 0, 45), (127, 0,75), (127, 0,102), (127, 0, 127), (127, 0, 151), (127, 0,173), (127, 0,195), (127, 0,215), (127, 0, 235)], [(151, 0, 45), (151, 0,75), (151, 0,102), (151, 0, 127), (151, 0, 151), (151, 0,173), (151, 0,195), (151, 0,215), (151, 0, 235)], [(173, 0, 45), (173, 0,75), (173, 0,102), (173, 0, 127), (173, 0, 151), (173, 0,173), (173, 0,195), (173, 0,215), (173, 0, 235)], [(195, 0, 45), (195, 0,75), (195, 0,102), (195, 0, 127), (195, 0, 151), (195, 0,173), (195, 0,195), (195, 0,215), (195, 0, 235)], [(215, 0, 45), (215, 0,75), (215, 0,102), (215, 0, 127), (215, 0, 151), (215, 0,173), (215, 0,195), (215, 0,215), (215, 0, 235)], [(235, 0, 45), (235, 0,75), (235, 0,102), (235, 0, 127), (235, 0, 151), (235, 0,173), (235, 0,195), (235, 0,215), (235, 0, 235)]]		#for reconstructed image

# if sys.argv[2]=="1":
# 	colours=colours1
# else:
# 	colours=colours2
	
boundaries=[]
for i in range(len(colours1)):
	for j in range(len(colours1[i])):
		tup = colours1[i][j]
		lower = np.array([color-s if color-s>-1 else 0 for color in tup], dtype="int16")
		upper = np.array([color+s if color+s<256 else 255 for color in tup], dtype="int16")
		boundaries.append( (lower,upper ) )
		# boundaries.append( ([tup[0]-s,tup[1]-s,tup[2]-s],[tup[0]+s,tup[1]+s,tup[2]+s] ) )


# print "Colours : ", colours
# print "Boundaries : ", boundaries
countOrg=0
coordsOrg=[]
for ind, (lower, upper) in enumerate(boundaries):
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask=mask)
	# cv2.imwrite(str(i) +sys.argv[1], output)
	output = np.array(output)
	elems = np.array([(i,j) for i in range(0,output.shape[0],10) for j in range(0,output.shape[1],10) if match_color(output[i,j],colours1[ind//9][ind%9]) ])
	# print elems
	x,y = -1,-1
	if len(elems)!=0:
		y,x= np.mean(elems,axis=0)
	else:
		countOrg = countOrg+1
	coordsOrg.append((x,y))
coordsOrg= np.array(coordsOrg)



boundaries=[]
for i in range(len(colours2)):
	for j in range(len(colours2[i])):
		tup = colours2[i][j]
		lower = np.array([color-s if color-s>-1 else 0 for color in tup], dtype="int16")
		upper = np.array([color+s if color+s<256 else 255 for color in tup], dtype="int16")
		boundaries.append( (lower,upper ) )

count=0
coords=[]
for ind, (lower, upper) in enumerate(boundaries):
	mask = cv2.inRange(imageReconstructed, lower, upper)
	output = cv2.bitwise_and(imageReconstructed, imageReconstructed, mask=mask)
	output = np.array(output)
	elems = np.array([(i,j) for i in range(0,output.shape[0],10) for j in range(0,output.shape[1],10) if match_color(output[i,j],colours2[ind//9][ind%9]) ])
	x,y = -1,-1
	if len(elems)!=0:
		y,x= np.mean(elems,axis=0)
	else:
		count = count+1
	coords.append((x,y))
coords= np.array(coords)

diff = coordsOrg- coords

# print diff
print "Difference : " np.sqrt(np.mean(np.square(diff)))
print "Could not find : ", countOrg, " points in original image"
print "Could not find : ", count, " points in reconstructed image"
