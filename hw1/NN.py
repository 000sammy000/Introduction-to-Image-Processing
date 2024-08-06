import cv2
import numpy as np


def nearest_neighbor_rotate(image, angle):

    angle = np.deg2rad(angle)*-1

    center_x = image.shape[1] / 2
    center_y = image.shape[0] / 2

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    rotated_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
          
            rel_x = x - center_x
            rel_y = y - center_y
       
            rotated_x, rotated_y = np.dot(rotation_matrix, [rel_x, rel_y])
        
            rotated_x += center_x
            rotated_y += center_y

            if 0 <= rotated_x < image.shape[1]-1 and 0 <= rotated_y < image.shape[0]-1:
                rotated_image[y, x] = image[np.around(rotated_y).astype(int), np.around(rotated_x).astype(int)]
    return rotated_image

def nearest_neighbor_resize(image, scale_factor):
 
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
    
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            
            original_i = min(np.around(i/scale_factor).astype(int),image.shape[0]-1)
            original_j = min(np.around(j/scale_factor).astype(int),image.shape[1]-1)
            
            resized_image[i, j] = image[original_i, original_j]
    return resized_image


img=cv2.imread('building.jpg')

angle=30
scale_factor = 2

rotated_image = nearest_neighbor_rotate(img, angle)
resized_image = nearest_neighbor_resize(img,scale_factor)

cv2.imwrite("NN_resize.jpg", resized_image)
cv2.imwrite("NN_rotate.jpg",rotated_image)


cv2.imshow('My Image',img)
cv2.imshow('Rotate',rotated_image)
cv2.imshow('Resize',resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()