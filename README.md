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

## Final results

In our experiments, we have evaluated the performance of several classical machine learning models and also multiple
 state-of-the-art deep learning models for time series classification. 
Each model was evaluated on 4 different publicly available datasets of reasonable size. 
For the evaluation metrics, the accuracy and weighted average accuracy over all classes were used. 
The overall best performance was achieved by the Random Forest classifier on denoised rhythm data (considering weighted average accuracy)
achieving almost state-of-the-art results for the given task.


Considering consistency over different datasets again Random Forest classifier seems to perform the best beating
multiple more complex deep learning models.
When it comes to the performance of the models across different datasets, we can see that models generally performed better on beat data compared to rhythm ones
but there are exceptions and even the best overall result was achieved on the rhythm data.
Comparing the denoised versus raw one for some models denoising seems to help but for others such as GaussianNB it makes it worse
compared with raw ones so we can't simply say if it helps or not.
For a detailed comparison check the table below where you can find exact evaluation results for all models and datasets.

|Dataset and model                   |accuracy          |macro avg         |weighted avg      |
|------------------------------------|------------------|------------------|------------------|
|ECGData_Rhythm_XCM                  |0.2903225806451613|0.2115053515053514|0.3112673964286868|
|ECGData_Beat_MiniRocket             |0.3052631578947368|0.1090877192982456|0.3541680517082179|
|ECGDataDenoised_Beat_TST            |0.631578947368421 |0.0958133971291865|0.5075295895240493|
|ECGDataDenoised_Rhythm_XCM          |0.3978494623655914|0.1850793650793651|0.3593241167434716|
|ECGDataDenoised_Beat_1D CNN         |0.4421052631578947|0.0980537042403168|0.4444376661735474|
|ECGDataDenoised_Beat_LSTM           |0.3684210526315789|0.0918803418803418|0.392442645074224 |
|ECGDataDenoised_Beat_KNN            |0.631578947368421 |0.0774193548387096|0.4889643463497453|
|ECGDataDenoised_Rhythm_KNN          |0.4623655913978494|0.1053921568627451|0.2923782416192283|
|ECGData_Beat_GaussianNB             |0.6210526315789474|0.094             |0.5042105263157896|
|ECGDataDenoised_Rhythm_InceptionTime|0.4946236559139785|0.3086106023606024|0.4813247535021728|
|ECGData_Rhythm_InceptionTime        |0.3655913978494624|0.1971626939368874|0.3427616434900514|
|ECGData_Beat_XCM                    |0.3578947368421052|0.1030194805194805|0.3756801093643198|
|ECGData_Beat_Random Forest          |0.6534325889164598|0.0897356270644602|0.5366401611460966|
|ECGDataDenoised_Rhythm_MiniRocket   |0.5268817204301075|0.3209472234986055|0.5216227142794894|
|ECGDataDenoised_Beat_Random Forest  |0.6210526315789474|0.0766233766233766|0.4839371155160628|
|ECGDataDenoised_Beat_SVM            |0.631578947368421 |0.0774193548387096|0.4889643463497453|
|ECGData_Rhythm_Random Forest        |0.4516129032258064|0.1037037037037037|0.2876941457586619|
|ECGData_Rhythm_1D CNN               |0.2688172043010752|0.12196935891715  |0.2295442222893964|
|ECGData_Beat_Random Forest          |0.5684210526315789|0.085375816993464 |0.4748194014447884|
|ECGData_Beat_KNN                    |0.631578947368421 |0.0774193548387096|0.4889643463497453|
|ECGData_Beat_InceptionTime          |0.4947368421052631|0.1759632690541781|0.4851013484123531|
|ECGData_Beat_SVM                    |0.631578947368421 |0.0774193548387096|0.4889643463497453|
|ECGDataDenoised_Beat_XCM            |0.3368421052631579|0.0819897084048027|0.3714182540399025|
|ECGData_Rhythm_GaussianNB           |0.3333333333333333|0.180896570551743 |0.3103156751655083|
|ECGData_Rhythm_KNN                  |0.4623655913978494|0.1053921568627451|0.2923782416192283|
|ECGData_Rhythm_MiniRocket           |0.3548387096774194|0.1943425119948417|0.365192997202478 |
|ECGDataDenoised_Rhythm_SVM          |0.5161290322580645|0.1911111111111111|0.3856630824372759|
|ECGDataDenoised_Rhythm_Random Forest|0.6560150375939849|0.3705734660240814|0.6060608752944004|
|ECGDataDenoised_Rhythm_TST          |0.4408602150537634|0.1858858858858859|0.3651457909522426|
|ECGDataDenoised_Rhythm_Random Forest|0.4301075268817204|0.1811077132178049|0.3621308705653185|
|ECGData_Rhythm_TST                  |0.4086021505376344|0.1808688387635756|0.332794351470073 |
|ECGData_Beat_TST                    |0.6210526315789474|0.1098434004474272|0.5148004238784881|
|ECGDataDenoised_Beat_InceptionTime  |0.5052631578947369|0.1028123249299719|0.4826890756302519|
|ECGData_Beat_1D CNN                 |0.231578947368421 |0.0841647241647241|0.2698441908968225|
|ECGDataDenoised_Rhythm_GaussianNB   |0.3118279569892473|0.148076923076923 |0.2747311827956989|
|ECGData_Rhythm_SVM                  |0.4731182795698925|0.1261297182349814|0.318216117027662 |
|ECGDataDenoised_Beat_GaussianNB     |0.6105263157894737|0.0768211920529801|0.4851864761240849|
|ECGDataDenoised_Beat_MiniRocket     |0.3473684210526316|0.1457886557886558|0.393583609373083 |
|ECGDataDenoised_Beat_Random Forest  |0.6650124069478908|0.0973467441815496|0.5483525323827746|
|ECGData_Rhythm_LSTM                 |0.1612903225806451|0.1083603896103896|0.1937374901084578|
|ECGDataDenoised_Rhythm_LSTM         |0.2580645161290322|0.2031450962368076|0.2660955812172456|
|ECGData_Rhythm_Random Forest        |0.5344611528822055|0.2782146696956576|0.4870055110658955|
|ECGData_Beat_LSTM                   |0.4               |0.0821848739495798|0.4095886775762937|
|ECGDataDenoised_Rhythm_1D CNN       |0.2795698924731182|0.1377998345740281|0.3081725766429201|



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
