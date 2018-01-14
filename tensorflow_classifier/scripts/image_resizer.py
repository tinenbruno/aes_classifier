import cv2
import math
import os

BLACK = [0, 0, 0]
IMAGES_INPUT_FOLDER = "/home/bruno/Documents/AVA/nature2/photos/positive"
IMAGES_OUTPUT_FOLDER = "/home/bruno/Documents/AVA/nature_resized/photos/positive"

def resize(input_image, size):
    height, width, channels = input_image.shape
    if height > width:
        result = cv2.resize(input_image, (math.floor(size/height * width), size), interpolation = cv2.INTER_AREA)
        padding = size - math.floor(size/height * width)
        left = math.floor(padding/2)
        right = padding - left
        return cv2.copyMakeBorder(result, 0, 0, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    else:
        result = cv2.resize(input_image, (size, math.floor(size/width * height)), interpolation = cv2.INTER_AREA)
        padding = size - math.floor(size/width * height)
        top = math.floor(padding/2)
        bottom = padding - top
        return cv2.copyMakeBorder(result,top,bottom,0,0,cv2.BORDER_CONSTANT,value = BLACK)

def main():
    for filename in os.listdir(IMAGES_INPUT_FOLDER):
        image = cv2.imread(IMAGES_INPUT_FOLDER + "/" + filename)
        resized_image = resize(image, 200)
        cv2.imwrite(IMAGES_OUTPUT_FOLDER + "/" + filename, resized_image)
        print(".", end="", flush=True)
        #print(IMAGES_OUTPUT_FOLDER + "/" + filename, end="", flush=True)

main()
