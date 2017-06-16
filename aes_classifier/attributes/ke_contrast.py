from ml_buff.models.base_feature_record import BaseFeatureRecord
import cv2

IMAGE_PATH = '/home/bruno/Downloads/images4AVA'

class KeContrast(BaseFeatureRecord):
    def calculate(self, input_data):
        image = cv2.imread('{0}/{1}.jpg'.format(IMAGE_PATH ,input_data.external_id))
        if (image is None):
            return [0]
        channels = cv2.split(image)
        colors = ("h", "s", "v")
        for (channel, color) in zip(channels, colors):
            if color == "h":
                histogram = cv2.calcHist([channel], [0], None, [8], [0, 180])
            else:
                histogram = cv2.calcHist([channel], [0], None, [8], [0, 256])

        return cv2.normalize(histogram, histogram)

