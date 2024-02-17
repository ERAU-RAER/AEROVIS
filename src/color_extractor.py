import cv2 as cv
import numpy as np

COLOR_RANGES = {
    "white": ((0, 0, 168), (172, 111, 255)),
    "black": ((0, 0, 0), (180, 255, 30)),
    "red1": ((0, 50, 20), (5, 255, 255)),
    "red2": ((175, 50, 20), (180, 255, 255)),
    "blue": ((110, 50, 50), (130, 255, 255)),
    "green": ((50, 50, 50), (70, 255, 255)),
    "purple": ((129, 50, 70), (158, 255, 255)),
    "brown": ((10, 100, 20), (20, 255, 200)),
    "orange": ((10, 100, 20), (25, 255, 255)),
}


def get_shape_color(src_img):
    hsv = cv.cvtColor(src_img, cv.COLOR_BGR2HSV)

    color_pixels = {color: 0 for color in COLOR_RANGES}

    for color, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv.inRange(hsv, lower, upper)
        
        # combine red masks
        if color == "red1" or color == "red2":
            red_mask = mask if color == "red1" else red_mask | mask
            if color == "red2": 
                color_pixels["red"] = np.sum(red_mask) / 255
        else:
            color_pixels[color] = np.sum(mask) / 255

    return max(color_pixels, key=color_pixels.get)
