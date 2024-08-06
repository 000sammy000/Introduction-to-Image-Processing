import cv2
import numpy as np

def histogram_equalization(image):
   
    height,width=image.shape[:2]
    MN=height*width
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist=np.zeros(256)
    for i in range(height):
        for j in range(width):
            hist[gray[i,j]]+=1

    cdf = np.zeros(256)
    for i in range(256):
        if i==0:
            cdf[i]=hist[0]/MN
        else:
            cdf[i]=cdf[i-1]+hist[i]/MN


    cdf = cdf*255
    cdf = cdf.astype('uint8')
    equalized = cdf[gray]
    
    return equalized


image = cv2.imread('Q1.jpg')

equalized_image = histogram_equalization(image)

cv2.imwrite('Q1_output.jpg', equalized_image)
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
