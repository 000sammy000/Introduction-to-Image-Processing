import cv2
import numpy as np

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
                    
    
    
    # Perform bicubic interpolation
    interpolated_value = np.zeros(3)
    
    for k in range(3):

        result = np.dot(v, np.dot(B, np.dot(M[:, :, k], np.dot(C, u))))
        interpolated_value[k] = np.clip(result, 0, 255).astype(np.uint8)
        
   
        
   
    
    return interpolated_value



def bicubic_resize(image, scale):

    h, w = image.shape[:2]
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    
    enlarged_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    
    for i in range(new_h):
        for j in range(new_w):
            
            x = i / scale
            y = j / scale
            enlarged_image[j,i]=bicubic_interpolation(image,y,x,h,w)
    
    return enlarged_image



input_image = cv2.imread('building.jpg')


scale = 2

enlarged_image = bicubic_resize(input_image, scale)


cv2.imwrite("bic_resize.jpg",enlarged_image)


cv2.imshow('Original Image', input_image)

cv2.imshow('Enlarged Image', enlarged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
