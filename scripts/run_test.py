from pathlib import Path
from typing import Optional, Dict, Union

import yaml
import fire

from alebrew.data.data import AtomicStructures
from alebrew.strategies import EvaluationStrategy
from alebrew.utils.config import update_config
from alebrew.model.forward import load_models_from_folder


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
        if 'test' not in split:
            raise RuntimeError(f'Current split does not contain test data. In this case make sure to provide '
                               f'a separate test data path.')
        test_structures = split['test']
    else:
        test_structures = AtomicStructures.from_file(config['test_data_path'], **config)

    # load trained models and evaluate them on the test set
    models = load_models_from_folder(config['model_path'], len(config['model_seeds']), key='best')
    evaluate = EvaluationStrategy(config)
    _, _ = evaluate.run(models, test_structures=test_structures, folder=config['model_path'])


if __name__ == '__main__':
    fire.Fire(main)
