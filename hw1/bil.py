import cv2
import numpy as np


def bilinear_interpolation(image, x, y):
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, image.shape[1] - 1), min(y0 + 1, image.shape[0] - 1)
    dx, dy = x - x0, y - y0
    interpolated_value = (1 - dx) * (1 - dy) * image[y0, x0] + \
                         dx * (1 - dy) * image[y0, x1] + \
                         (1 - dx) * dy * image[y1, x0] + \
                         dx * dy * image[y1, x1]
    return interpolated_value


def bilinear_resize(image, scale_factor):

    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)
  
    resized_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            original_i = i / scale_factor
            original_j = j / scale_factor
            resized_image[i, j] = bilinear_interpolation(image, original_j, original_i)
    return resized_image


def rotate_image(image, angle):

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
            if 0 <= rotated_x < image.shape[1] and 0 <= rotated_y < image.shape[0]:
                rotated_image[y, x] = bilinear_interpolation(image, rotated_x, rotated_y)
            """else:
                rotated_image[y, x] = [0, 0, 0]"""
    return rotated_image


img=cv2.imread('building.jpg')

scale_factor = 2

bilinear_image=bilinear_resize(img,scale_factor)
bilinear_rotate=rotate_image(img,30)


cv2.imwrite('bil_resized.jpg',bilinear_image)
cv2.imwrite('bil_rotate.jpg',bilinear_rotate)



cv2.imshow('My Image',img)
cv2.imshow('Bilinear Resize',bilinear_image)
cv2.imshow('Bilinear Rotate',bilinear_rotate)

cv2.waitKey(0)
cv2.destroyAllWindows()