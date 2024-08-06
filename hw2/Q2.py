import numpy as np
import cv2

def histogram_specification(source_image, target_image):
 
    height,width=source_image.shape[:2]
    MN=height*width
    
    src_hist=np.zeros(256)
    tar_hist=np.zeros(256)
    for i in range(height):
        for j in range(width):
            src_hist[source_image[i,j]]+=1
            tar_hist[target_image[i,j]]+=1

    source_cdf=np.zeros(256)
    target_cdf=np.zeros(256)
    for i in range(256):
        if i==0:
            source_cdf[i]=src_hist[0]/MN
            target_cdf[i]=tar_hist[0]/MN
        else:
            source_cdf[i]=(source_cdf[i-1]+src_hist[i]/MN)
            target_cdf[i]=(target_cdf[i-1]+tar_hist[i]/MN)
    
    spec_mapping = np.zeros(256)
    for i in range(256):
        spec_mapping[i] = np.argmin(np.abs(source_cdf[i] - target_cdf))

    
    spec_image = spec_mapping[source_image]
    
    return spec_image.astype('uint8')


if __name__ == "__main__":
    source_image = cv2.imread('Q2_source.jpg', cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread('Q2_reference.jpg', cv2.IMREAD_GRAYSCALE)

    spec_image = histogram_specification(source_image, target_image)
    
    cv2.imshow('Source Image', source_image)
    cv2.imshow('Target Image', target_image)
    cv2.imshow('Specified Image', spec_image)
    cv2.imwrite('Q2_output.jpg', spec_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
