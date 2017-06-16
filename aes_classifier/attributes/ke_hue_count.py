from ml_buff.models.base_feature_record import BaseFeatureRecord
import cv2

IMAGE_PATH = '/home/bruno/Downloads/images4AVA'

class KeHueCount(BaseFeatureRecord):
    def calculate(self, input_data):
        image = cv2.imread('{0}/{1}.jpg'.format(IMAGE_PATH ,input_data.external_id))
        if (image is None):
            return [0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv)
        h_hist = cv2.calcHist(hsv, [0], None, [20], [0, 180])
        s_hist = cv2.calcHist(hsv, [1], None, [20], [0, 256])
        cv2.normalize(s_hist, s_hist)
        v_hist = cv2.calcHist(hsv, [2], None, [20], [0, 256])
        cv2.normalize(v_hist, v_hist)

        return cv2.normalize(histogram, histogram)

