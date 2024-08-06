import cv2
import numpy as np
from scipy.ndimage import map_coordinates
import math

def rotate_bicubic(image, angle):
    
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)*(-1)
    
    

    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)],])
    
   
    center_x = h/ 2
    center_y = w / 2
    
    output_shape = image.shape
  
    output_image = np.zeros_like(image)
    
  
    for x in range(h):
        for y in range(w):
         
            rel_x = x - center_x
            rel_y = y - center_y
            rotated_x, rotated_y = np.dot(rotation_matrix, [rel_x, rel_y])
            rotated_x += center_x
            rotated_y += center_y
            
           
            if 0 <= rotated_x < image.shape[1] - 1 and 0 <= rotated_y < image.shape[0] - 1:
                output_image[y, x] = bicubic_interpolation(image, rotated_y, rotated_x,h,w)
    
    return output_image

def bicubic(p1, p2, p3, p4, distance):
    f0=p2
    f1=p3
    df0=(p3-p1)/2
    df1=(p4-p2)/2
    
    return (2 * f0 - 2 * f1 +  df0 + df1) * (distance ** 3) + \
        (-3* f0 - 3 * f1 - 2 * df0 -  df1) * \
        (distance ** 2) + (df0) * distance + df1



def overflow_address(color_value):
    if color_value > 255:
        return 255
    elif color_value < 0:
        return 0
    else:
        return color_value


def bicubic_interpolation(image, y, x,h,w):
    
   
    

    x2 = int(np.floor(x))
    y2 = int(np.floor(y))
    x1 = max(x2 - 1, 0)
    y1 = max(y2 - 1, 0)
    x3 = min(x2 + 1, h - 1)
    y3 = min(y2 + 1, w - 1)
    x4 = min(x2 + 2, h - 1)
    y4 = min(y2 + 2, w - 1)

    dx = x - x2
    dy = y - y2
    u = np.array([1, dx, dx ** 2, dx ** 3])
    v = np.array([1, dy, dy ** 2, dy ** 3])
    
    F = np.array([[0,1,0,0],[0,0,1,0],[-1/2,0,1/2,0],[0,-1/2,0,1/2]])
    Fp = np.array([[0,0,-1/2,0],[1,0,0,-1/2],[0,1,1/2,0],[0,0,0,1/2]])
    B = np.array([[0, 1, 0, 0], [-1/2, 0, 1/2, 0], [1, -5/2, 2, -1/2], [-1/2, 3/2, -3/2, 1/2]])
    C = np.array([[0, -1/2, 1, -1/2], [1, 0, -5/2, 3/2], [0, 1/2, 2, -3/2], [0, 0, -1/2, 1/2]])
    
    M = np.array([[image[y1, x1], image[y1, x2], image[y1, x3], image[y1, x4]],
                [image[y2, x1], image[y2, x2], image[y2, x3], image[y2, x4]],
                [image[y3, x1], image[y3, x2], image[y3, x3], image[y3, x4]],
                [image[y4, x1], image[y4, x2], image[y4, x3], image[y4, x4]]])
                    
    
    
    interpolated_value = np.zeros(3)
    
    for k in range(3):
        result = np.dot(v, np.dot(B, np.dot(M[:, :, k], np.dot(C, u))))

        interpolated_value[k] = np.clip(result, 0, 255).astype(np.uint8)
        
   
    
    return interpolated_value

image = cv2.imread('building.jpg')

rotated_image = rotate_bicubic(image, 30)

cv2.imwrite("bic_rotate.jpg",rotated_image)

cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
