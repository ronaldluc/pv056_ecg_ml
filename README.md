# pv056_ecg_ml
PV056 Machine Learning and Data Mining project

Test before pushing a commit that this line produces all the results on all the datasets:
```bash
docker build -t pv056_ecg_ml docker
docker run -it --rm -v $PWD/src:/home/work/src -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py  
```
Please check reasonable results were produced in the `output` folder.

## Partial black-box testing
To run without downloading the data again:
```bash
docker run -it --rm  -v $PWD/src:/home/work/src -v $PWD/data/downloaded:/home/work/data/downloaded -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py
```

To run with the precomputed information:
```bash
docker run -it --rm -v $PWD/src:/home/work/src -v $PWD/data:/home/work/data -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py
```

## Final results

In our experiments, we have evaluated the performance of several classical machine learning models and also multiple state-of-the-art deep learning models for time series classification. Each model was evaluated on 3 different publicly available datasets of reasonable size. For the evaluation metrics, the f score and accuracy were used.
The overall best performance was achieved by TODO achieving almost state-of-the-art results for the given task.
For a detailed comparison check the table below where you can find exact evaluation results for all models.


## Reproduce results

Without GPU
```bash
git clone https://github.com/ronaldluc/pv056_ecg_ml.git
cd pv056_ecg_ml
docker build -t pv056_ecg_ml docker
docker run -it --rm -v $PWD/src:/home/work/src -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py  
```

With CUDA GPU

```bash
git clone https://github.com/ronaldluc/pv056_ecg_ml.git
cd pv056_ecg_ml
docker build -t pv056_ecg_ml --build-arg FROM_IMAGE=tensorflow/tensorflow:latest-gpu docker
docker run -it --rm -v $PWD/src:/home/work/src --gpus=all -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py  
docker run -it --rm -v $PWD/src:/home/work/src --gpus=all  -v $PWD/data:/home/work/data -v $PWD/output:/home/work/output pv056_ecg_ml python3 main.py

```
