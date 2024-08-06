import cv2
import numpy as np

image = cv2.imread('dog.jpg', cv2.IMREAD_GRAYSCALE)
height, width = image.shape
print(height,width)
laplacian_filter = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

padding_image = np.zeros((height + 2, width + 2), dtype=np.uint8)

for i in range(0,height+2):
    temp_i=i
    if i==0:
        temp_i=i+1
    if i==height+1:
        temp_i=i-1
    for j in range(0,width+2):
        temp_j=j
        if j==0:
            temp_j=j+1
        if j==width+1:
            temp_j=j-1
        padding_image[i,j]=image[temp_i-1,temp_j-1]

laplacian_image=np.zeros((height, width), dtype=np.uint8)

for i in range(1,height+1):
    for j in range(1,width+1):
        roi = padding_image[i-1:i+2, j-1:j+2]
        filtered_pixel = np.sum(roi * laplacian_filter)
        laplacian_image[i-1, j-1] = max(0, min(255, filtered_pixel))
        

cv2.imshow('Laplacian Filtered Image',laplacian_image)
cv2.imwrite('spatial.jpg', laplacian_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
