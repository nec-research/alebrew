from pathlib import Path
from typing import Optional, Dict, Union

import yaml
import fire

from alebrew.data.data import AtomicStructures
from alebrew.model.forward import load_models_from_folder
from alebrew.strategies import TrainingStrategy
from alebrew.utils.config import update_config


def main(config: Optional[Union[str, Dict]] = None):
    
    # load config from config_file
    if isinstance(config, str):
        config = yaml.safe_load(Path(config).read_text())

    # update default config with provided parameters (is done twice as here we need to update data path)
    config = update_config(config)

    # manage data
    if config['data_path']:
        atomic_structures = AtomicStructures.from_file(config['data_path'], **config)
        split = atomic_structures.random_split({'train': config['n_train'], 'valid': config['n_valid']},
                                               seed=config['data_seed'])
    else:
        train_structures = AtomicStructures.from_file(config['train_data_path'], **config)
        valid_structures = AtomicStructures.from_file(config['valid_data_path'], **config)
        split = {'train': train_structures, 'valid': valid_structures}

    # load pretrained models if provided
    if config['pretrained_model_path']:
        pretrained_models = load_models_from_folder(config['pretrained_model_path'], len(config['model_seeds']),
                                                    key='best')
    else:
        pretrained_models = None

    # run training (fine-tuning)
    training = TrainingStrategy(config)
    _ = training.run(train_structures=split['train'], valid_structures=split['valid'], folder=config['model_path'],
                     pretrained_models=pretrained_models)


if __name__ == '__main__':
    fire.Fire(main)
