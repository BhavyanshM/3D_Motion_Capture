import cv2
import numpy as np
import glob
from scipy import linalg
from matplotlib import pyplot as plt

# np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# cap = cv2.VideoCapture(1)
# ret, img = cap.read()

def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
	lines - corresponding epilines '''
	print(img1.shape)
	r,c,h = img1.shape
	# img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	# img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	
	return img1,img2

n = 8

i = 0
width, height = 1024, 768

img1 = cv2.imread("frame_1.jpg")
img2 = cv2.imread("frame_2.jpg")

orb = cv2.ORB_create(nfeatures=3000)

feature = orb


kp1, des1 = feature.detectAndCompute(img1, None)
kp2, des2 = feature.detectAndCompute(img2, None)

# img1 = cv2.drawKeypoints(img1, kp1, None)
# img2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n], None)


A = np.zeros((n,9))

pts1 = []
pts2 = []

for m in matches[:n]:
	p1, p2 = kp1[m.queryIdx].pt, kp2[m.trainIdx].pt
	pts1.append(p1)
	pts2.append(p2)
	x1 = np.array([p1[0], p1[1], 1])
	x2 = np.array([p2[0], p2[1], 1])
	print("x1:", x1, "x2:", x2, "kron:", np.kron(x1, x2))
	A[i] = np.kron(x1, x2)
	i+=1


U, S, V = np.linalg.svd(A, full_matrices=False)
F = V[-1].reshape(3,3)

U,S,V = linalg.svd(F)
S[2] = 0
F = np.dot(U,np.dot(np.diag(S),V))
F = F/F[2,2]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

FundMat, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
print(FundMat)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


print(F)


# cv2.namedWindow("Image 1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image 1', width,height)
# cv2.imshow('Image 1', img1)

# cv2.namedWindow("Image 2", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image 2', width,height)
# cv2.imshow('Image 2', img2)


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,FundMat)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,FundMat)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

# cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Matches', 2*width,height)
# cv2.imshow('Matches', matching_result)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

# if cv2.waitKey(1) & 0xFF == ord('q'):
# 	cap.release()
# 	cv2.destroyAllWindows()
	# break

