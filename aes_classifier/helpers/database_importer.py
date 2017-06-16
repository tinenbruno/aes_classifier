from ml_buff.models.input_data import InputData
from ml_buff.database_helper import create_tables, drop_tables
from ml_buff.models.base_model import database

DATABASE = {
        'drivername': 'postgresql',
        'host': 'localhost',
        'port': '5432',
        'username': 'postgres',
        'password': 'postgres',
        'database': 'ml_buff'
        }

DATASET_DEFINITIONS = r'../../AVA_dataset/AVA.txt'

drop_tables()
create_tables()

file = open(DATASET_DEFINITIONS)
data_source = []


for line in file:
    line = line.strip().split(' ')
    data_source.append({ 'external_id': line[1], 'dataset_name': 'AVA' })

print('datasource built with {0} entries'.format(len(data_source)))

with database.atomic():
    for idx in range(0, len(data_source), 100):
        InputData.insert_many(data_source[idx:idx+100]).execute()
