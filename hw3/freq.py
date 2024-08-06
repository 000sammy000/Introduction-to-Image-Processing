import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape



H = np.zeros((height,width), dtype=np.float32)
for u in range(height):
    for v in range(width):
        H[u,v] = -4*np.pi*np.pi*((u-height/2)**2 + (v-width/2)**2)


spectrum = np.fft.fft2(image)
shift_spectrum = np.fft.fftshift(spectrum)


laplacian_image=H*shift_spectrum


f_ishift_1 = np.fft.ifftshift(laplacian_image)
img_1_back = np.fft.ifft2(f_ishift_1).real


OldRange = np.abs(np.min(img_1_back))
if np.abs(np.max(img_1_back))>np.abs(np.min(img_1_back)):
    OldRange = np.max(img_1_back)

NewRange = 255
LapScaled = (((img_1_back) * NewRange) / OldRange)


final_image=image-LapScaled
final_image = np.clip(final_image, 0, 255)
final_image = np.round(final_image).astype(np.uint8)


cv2.imshow('Laplacian Filtered Image',final_image)
cv2.imwrite('freq.jpg', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


