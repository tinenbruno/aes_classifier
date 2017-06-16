from ml_buff.models.base_feature_record import BaseFeatureRecord
import cv2

IMAGE_PATH = '/home/bruno/Downloads/images4AVA'

class KeBrightness(BaseFeatureRecord):
    def calculate(self, input_data):
        image = cv2.imread('{0}/{1}.jpg'.format(IMAGE_PATH, input_data.external_id))
        if (image is None):
            return [0]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        value = cv2.split(hsv)[2]
        return value.mean()

