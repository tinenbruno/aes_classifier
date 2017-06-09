from ml_buff.models.input_data import InputData
from ml_buff.helpers.feature_value_helper import FeatureValueHelper
from ml_buff.models.base_input_data_repository import BaseInputDataRepository

from attributes.ke_contrast import KeContrast

DATASET_IMAGES_PATH = r'../../AVA_dataset/image'

input_ids = BaseInputDataRepository().getAllForDataset('AVA')

input_data = []

for i in range(0, 255530):
    input_data.append({'input_id': input_ids[i].id, 'values': i})

print('dataset loaded with {0} instances'.format(len(input_data)))
print('dataset loaded with {0} instances'.format(input_data[10]))

FeatureValueHelper.createAll(input_data)
