from ml_buff.models.base_feature_record import BaseFeatureRecord
import cv2
import numpy as np

IMAGE_PATH = '/home/bruno/Downloads/images4AVA'

class KeHueCount(BaseFeatureRecord):
    def calculate(self, input_data):
        image = cv2.imread('{0}/{1}.jpg'.format(IMAGE_PATH ,input_data.external_id))
        if (image is None):
            return [0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = [0] * 20 #20 bin histogram
        for i in range(0, hsv.size):
            for j in range(0, hsv[i].size):
                if hsv[i][j][1] >= 51 and hsv[i][j][2] >= 38 and hsv[i][j] <= 243:
                    histogram[hsv[i][j][0]%20] = histogram[hsv[i][j][0]%20] + 1

        max = np.amax(histogram)
        alpha = 0.2

        threshold = max * alpha

        N = 0
        for i in range(0, len(histogram)):
            if histogram[i] > threshold:
                N = N + 1
        return 20 - N

