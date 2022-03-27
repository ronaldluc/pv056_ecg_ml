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