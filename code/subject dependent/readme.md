## 🔨 How to run the code?
Please follow the instructions:<br/>

1. Data Handling and Preprocessing<br/>
Here **subject** varies from s01 to s32.
```
python data_preprocessing.py --subject [subject_name] --deap_dataset_path [deap dataset path] --datafiles_path [where you want to save the .csv files]
```

2. HSFCS based channel selction<br/>
```
python channel_selection.py --subject [subject name] --datafiles_path [put the path where the subject wise .csv files kept]
```

3. Discreate Wavelet based feature extraction<br/>
```
python wavelet_features.py --subject [subject_name] --deap_dataset_path [deap dataset path] --datafiles_path [where you want to save the .csv files]
```

4. Genetic Algorithm based feature selection<br/>
```
python GA_feature_selection.py --subject [subject name] --datafiles_path [put the path where the subject wise .csv files kept]
```