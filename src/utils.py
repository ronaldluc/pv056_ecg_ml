import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Iterable
import wfdb
import ast
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def setup_logger(name: str):
    logging.getLogger(name).setLevel(logging.INFO)
    logging.getLogger(name).addHandler(logging.StreamHandler(stream=sys.stdout))
    logging.getLogger(name).handlers[0].setFormatter(
        logging.Formatter('[%(name)s] [%(levelname)s]:\t%(message)s')
    )


def download_files(urls: List[str], path: Path):
    path.mkdir(exist_ok=True, parents=True)
    for url in urls:
        file_path = path / url.split('/')[-1]
        zip_path = file_path.with_suffix('.zip')
        if not zip_path.exists():
            with open(zip_path, 'wb') as f:
                f.write(requests.get(url).content)
            os.system(f'unzip {zip_path} -d {path}')

    # TODO: Download other datasets


def load_raw_data(df, sampling_rate, path: Path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def preprocess_datasets(in_folder: Path, out_folder: Path, cache=True):
    """
    Preprocess the datasets in the input folder and save them
    in the output folder as numpy arrays.
    """
    odf = pd.read_excel(in_folder / '15653771.zip')
    head = None
    ldf = odf.drop(index=1789)  # broken row
    ldf = ldf.iloc[:head]
    datasets = ['ECGData', 'ECGDataDenoised']

    for dataset_name in datasets:
        logging.getLogger(__name__).info(f'Working on {dataset_name}')
        if (out_folder / dataset_name / 'x.npy').exists() and cache:
            logging.getLogger(__name__).info(f'\tAlready processed')
            continue
        cum = []
        dataset_folder = in_folder / dataset_name
        for file_name in tqdm(ldf['FileName']):  # overengineer to parallelism
            cum.append(pd.read_csv(dataset_folder / f'{file_name}.csv').interpolate().values)
        x = np.array(cum)
        (out_folder / dataset_name).mkdir(exist_ok=True, parents=True)

        y = ldf['Rhythm'].values
        logging.getLogger(__name__).info(f'{x.dtype} {x.shape}')
        logging.getLogger(__name__).info(f'{y.dtype} {y.shape}')
        np.save(out_folder / dataset_name / 'x.npy', x)
        np.save(out_folder / dataset_name / 'y.npy', y)

    # TODO: Preprocess the other datasets
    Y = pd.read_csv(Path + 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'+ 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    sampling_rate = 100
    X = load_raw_data(Y, sampling_rate, Path)
    Y['diagnostic_superclass'] = Y.scp_codes.apply('aggregate_diagnostic')
    np.save(out_folder / dataset_name / 'X.npy', X)
    np.save(out_folder / dataset_name / 'Y.npy', Y)


def print_results(results_folder: Path, output_folder: Path):
    logging.getLogger(__name__).info('Results')
    res = []
    for file_name in results_folder.glob('*.csv'):
        ser = pd.read_csv(file_name, index_col=0).loc[
            ['accuracy', 'macro avg', 'weighted avg'], ['f1-score']].squeeze()
        ser.name = '_'.join(file_name.name.split('_')[:-1])
        res.append(ser)
    df = pd.DataFrame(res)
    for line in str(df).split('\n'):
        logging.getLogger(__name__).info(f'\t\t{line}')

    formatted_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    df.to_csv(output_folder / f'results_{formatted_time}.csv')

    # TODO: pivot, plot, etc.
    # TODO: statistically analyze the results (e.g. t-test, wilcoxon, etc.)


setup_logger(__name__)


def to_basics_dict(obj, max_depth=5):
    if max_depth <= 0:
        return '...'
    try:
        if isinstance(obj, dict):
            return {k: to_basics_dict(v, max_depth - 1) for k, v in obj.items()}
        if isinstance(obj, Iterable):
            return [to_basics_dict(v, max_depth - 1) for v in obj]
        if isinstance(obj, (int, float, str)):
            return obj
        if hasattr(obj, '__dict__'):
            return to_basics_dict(obj.__dict__, max_depth - 1)
    except TypeError:
        return None


def assign_dataset(df_):
    len_ = len(df_)
    df_ = df_.index.to_frame()
    df_['dataset'] = 'train'
    df_.loc[df_.iloc[int(0.7 * len_):int(0.85 * len_)].index, 'dataset'] = 'dev'
    df_.loc[df_.iloc[int(0.85 * len_):].index, 'dataset'] = 'test'
    return df_
