import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

img = cv2.imread('lena_gray.png')

# Part a : 2D convolution using Sobel filter

# Starting clock for 2D convolution with Sobel Filters
start_2d = time.clock()

# Declaring Sobel Filters
sobel_kernel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_kernel_y = np.array([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]])


# Computing Gradient Image Gx in X direction
grad_x = cv2.filter2D(img,-1,sobel_kernel_x)

# Computing Gradient Image Gy in Y direction
grad_y = cv2.filter2D(img,-1,sobel_kernel_y)

# Way 1 to generate magnitude G = √G x2 + G y2 from Gradient Images Gx and Gy
# grad = np.rint(np.sqrt(np.power(grad_x, 2)+ np.power(grad_y, 2))).astype(int)

# Way 2 to generate magnitude G = √G x2 + G y2 from Gradient Images Gx and Gy
grad = np.hypot(grad_x, grad_y)

# Normalizing the output of Gradient G
grad = (grad - np.min (grad))/ (np.max(grad)- np.min(grad))
grad = np.rint(grad*255.0) 

# Ending clock for 2D convolution with Sobel Filters
end_2d = time.clock() - start_2d

######################################################################

# Part b : 1D convolution using Separable Sobel filter

# Starting clock for 1D convolution with Separable Sobel Filters
start_1d = time.clock()

# Declaring Sobel Separable Filters
sobel_sep_kernel_x_1 = np.array([[ 1],
                                 [ 2],
                                 [ 1]])

sobel_sep_kernel_x_2 = np.array([[-1, 0, 1]])                    

sobel_sep_kernel_y_1 = np.array([[-1],
                                 [ 0],
                                 [ 1]])

sobel_sep_kernel_y_2 = np.array([[ 1, 2, 1]])                    

# Computing Gradient Image Gx in X direction
grad_sep_x = cv2.sepFilter2D(img, -1, sobel_sep_kernel_x_2, sobel_sep_kernel_x_1)

# Computing Gradient Image Gy in Y direction
grad_sep_y = cv2.sepFilter2D(img, -1, sobel_sep_kernel_y_2, sobel_sep_kernel_y_1)

# Generating magnitude G = √G x2 + G y2 from Gradient Images Gx and Gy
grad_sep = np.rint(np.sqrt(np.power(grad_sep_x, 2)+ np.power(grad_sep_y, 2))).astype(int)

# Ending clock for 1D convolution with Separable Sobel Filters
end_1d = time.clock() - start_1d

######################################################################

# Part c : Printing results
print("Time taken for 2D convolution with Sobel Filters : ")
print(end_2d)

print("Time taken for 1D convolution with Separable Sobel Filters : ")
print(end_1d)

# Part a and b : Plotting output Images

# Part a : Result after 2D convolution
plt.subplot(331),plt.imshow(grad_x),plt.title('Gradient Image Gx')
plt.xticks([]), plt.yticks([])

plt.subplot(332),plt.imshow(grad_y),plt.title('Gradient Image Gy')
plt.xticks([]), plt.yticks([])

plt.subplot(333),plt.imshow(grad),plt.title('Gradient')
plt.xticks([]), plt.yticks([])

# Part b : Result after 1D convolution
plt.subplot(334),plt.imshow(grad_sep_x),plt.title('Gradient Sep Gx')
plt.xticks([]), plt.yticks([])

plt.subplot(335),plt.imshow(grad_sep_y),plt.title('Gradient Sep Gy')
plt.xticks([]), plt.yticks([])

plt.subplot(336),plt.imshow(grad_sep),plt.title('Gradient Sep')
plt.xticks([]), plt.yticks([])

plt.show()

# Observation : Result after 1D convolution is same as the one obtained from 2D convolution