from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.model.general_recommender.simrec import SimRec

config_file_list = ["config_simrec.yaml"]
config = Config(
    model=SimRec,
    dataset='goodreads',
    config_file_list=config_file_list
)
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)
model = SimRec(config, dataset).to(config['device'])

trainer = Trainer(config, model)
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
test_result = trainer.evaluate(test_data)
print(test_result)
