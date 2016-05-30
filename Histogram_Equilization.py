import cv2
import numpy as np
from matplotlib import pyplot as plt
import Image

# Reading color Image and converting it to a Grayscale Image
colorImg = cv2.imread("color.jpg")
grayImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg',grayImg)

# Reading the converted Grayscale Image
grayImg = cv2.imread('gray.jpg')
origImg = Image.open('gray.jpg')
pix = origImg.load()

# Step 1 : Initializing N, M, G and H
N = origImg.size[0]
M = origImg.size[1]
G = 256
hist = [0] * G

# Step 2 : Calculate the Histogram (H) values
for x in range(0, N):
    for y in range(0, M):
        hist[pix[x,y]] = hist[pix[x,y]] + 1


# Step 3 : Generating Cumulative Histogram (Hc) from the Histogram (H)
cumHist = [0] * G
cumHist[0] = hist[0]
for p in range(1, G):
    cumHist[p] = cumHist[p-1] + hist[p]
    

# Step 4 : Generating Lookup Table (Tp) as per the forumula in Algorithm 5.1
Tp = [0] * G
for p in range(0, G):
    Tp[p] = np.rint(((G-1)*cumHist[p])/(N*M))


# Step 5 : Rescanning the Original Image and generating Output Image from Lookup Table (Tp)
equMatrix = np.zeros((M,N)) 
for x in range(0, N):
    for y in range(0, M):
       equMatrix[y,x] = Tp[pix[x,y]]


equImg = Image.fromarray(equMatrix)
cv2.imwrite("equImg.jpg", equMatrix)

#########################################################

# Printing results

# Generating Cumulative Histogram values for Transformation Function (Tp)
cumHistTp = [0] * G
cumHistTp[0] = Tp[0]
for p in range(1, G):
    cumHistTp[p] = cumHistTp[p-1] + Tp[p]


plt.subplot(331),plt.imshow(colorImg),plt.title('Color Image')
plt.xticks([]), plt.yticks([])

plt.subplot(332),plt.imshow(grayImg),plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

plt.subplot(333),plt.imshow(equImg),plt.title('Equalized Image')
plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.hist(grayImg.ravel(),256,[0,256])
plt.title('Original Histogram')

plt.subplot(335),plt.plot(cumHist)
plt.title('Cumulative Histogram')

plt.subplot(336),plt.hist(equMatrix.ravel(),256,[0,256])
plt.title('Equalized Histogram')

plt.subplot(337),plt.plot(Tp)
plt.title('Transformation Function')

plt.show()
