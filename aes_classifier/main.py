from ml_buff.models.input_data import InputData
from ml_buff.models.feature_value import FeatureValue
from ml_buff.helpers.feature_value_helper import FeatureValueHelper
from ml_buff.models.base_input_data_repository import BaseInputDataRepository
from ml_buff.models.base_feature_repository import BaseFeatureRepository

from attributes.ke_contrast import KeContrast

DATASET_IMAGES_PATH = r'../../AVA_dataset/image'
IMAGE_PATH = '/home/bruno/Downloads/images4AVA'

input_ids = BaseInputDataRepository().getAllForDataset('AVA')

for i in range(0, 255530, 100):
    input_data = []
    for j in range(i, i+100):
        input_instance = BaseInputDataRepository().get(input_ids[j].id)
        feature = BaseFeatureRepository().get("KeContrast")
        value = KeContrast().calculate(input_instance)
        try:
            value = value.tolist()
        except:
            pass
        input_data.append({'value': value, 'input_data': input_instance, 'feature': feature})

    FeatureValue.insert_many(input_data).execute()
    print(i)

print('dataset loaded with {0} instances'.format(len(input_data)))
print('dataset loaded with {0} instances'.format(input_data[10]))

# FeatureValueHelper.createAll(input_data)
laplacian = np.array([[2/12, 8/12, 2/12], [8/12, -3, 8/12], [2/12, 8/12, 2/12]])

for i in range(0, 255530):
    image = cv2.imread('{0}/{1}.jpg'.format(IMAGE_PATH, input_ids[j].id))
    dst = cv2.filter2D(image, -1, laplacian)

