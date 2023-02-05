## 🔨 How to run the code?
Please follow the instructions:<br/>

1. Data Handling and Preprocessing (Here **subject** varies from s01 to s32):
```
python data_preprocessing.py --subject [subject name] --deap_dataset_path [deap dataset path] --datafiles_path [where you want to save the .csv files]
```

2. HSFCS based channel selction:<br/>
```
python channel_selection.py --subject [subject name] --datafiles_path [put the path where the subject wise .csv files are kept]
```

3. Discreate Wavelet based feature extraction:<br/>
```
python wavelet_features.py --subject [subject name] --deap_dataset_path [deap dataset path] --datafiles_path [where you want to save the .csv files]
```

4. Genetic Algorithm based feature selection:<br/>
```text
python GA_feature_selection.py \
--subject [subject name] \
--datafiles_path [put the path where the subject wise .csv files are kept] \
--alpha [weight for classification accuracy] \
--numPop [size of the population in GA] \
--numGen [No of generations in GA] \
--cxpb [crossover probability in GA] \
--mutpb [mutation probability in GA] \
--tournsize [tournsize in GA]
```
