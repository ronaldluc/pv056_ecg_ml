import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import ronald
import sunil
import adam
import martin
import bara
from utils import print_results, setup_logger, to_basics_dict, assign_dataset, download_files, preprocess_datasets


def evaluate_datasets(in_folder: Path, results_folder: Path, head=None):
    results_folder.mkdir(exist_ok=True, parents=True)
    for dataset in in_folder.glob('*'):
        logging.getLogger(__name__).info(f'Working on {dataset}')
        y_df = pd.Series(np.load(dataset / 'y.npy', allow_pickle=True))[:head]
        x = np.load(dataset / 'x.npy')[:head]
        print(f'x: {x.shape} y: {y_df.shape}')
        print(f'{y_df.value_counts()}')
        df_split = y_df.groupby(y_df).apply(assign_dataset)

        ids = {type_: df_split[df_split.dataset == type_].index for type_ in ['train', 'dev', 'test']}

        logging.getLogger(__name__).info({type_: len(y_df.loc[ids_]) for type_, ids_ in ids.items()})
        # for convenience
        train = ids['train']
        dev = ids['dev']
        test = ids['test']

        mlb = MultiLabelBinarizer(sparse_output=True)
        multilabel_y = y_df.values[:, None]
        y = pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(multilabel_y),
            index=y_df.index,
            columns=mlb.classes_).values

        models = [
            ronald.MLClassificator(),
            sunil.MLClassificator(),
            adam.MLClassificator(),
            bara.MLClassificator(),
            martin.MLClassificator(),
            # DL CLF
            ronald.DLClassificator(),
            sunil.XCMClassificator(),
            adam.LSTMClassificator(),
            bara.TSTClassificator(),
            martin.MiniRocketClassificator(),
            martin.InceptionTimeClassificator(),
        ]

        for model in models:
            hexdigest = hashlib.md5(str(to_basics_dict(model.__dict__)).encode('utf-8')).hexdigest()
            eval_name = f'{dataset.name}_{model.name}_{hexdigest}.csv'
            if (results_folder / eval_name).exists():
                continue
            print(str(model.__dict__).encode('utf-8'))
            logging.getLogger(__name__).info(f'\t{model.name}')
            model.fit(x[train], y[train], validation_data=(x[dev], y[dev]))
            y_pred = model.predict(x[test])
            for line in classification_report(y[test].argmax(-1), y_pred, target_names=mlb.classes_,
                                              zero_division=0).split('\n'):
                logging.getLogger(__name__).info(f'\t\t{line}')
            res = classification_report(y[test].argmax(-1), y_pred, target_names=mlb.classes_, zero_division=0,
                                        output_dict=True)
            pd.DataFrame(res).T.to_csv(results_folder / eval_name)


if __name__ == '__main__':
    urls = [
        'https://figshare.com/ndownloader/files/15651326',
        'https://figshare.com/ndownloader/files/15652862',
        'https://figshare.com/ndownloader/files/15653771',
    ]

    setup_logger(__name__)
    root_folder = Path('..')
    data_folder = root_folder / 'data'
    download_files(urls, data_folder / 'downloaded')
    preprocess_datasets(data_folder / 'downloaded', data_folder / 'processed')
    evaluate_datasets(data_folder / 'processed', root_folder / 'data' / 'intermediate_results', 4796)
    print_results(root_folder / 'data' / 'intermediate_results', root_folder / 'output')
