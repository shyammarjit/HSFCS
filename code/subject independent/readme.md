## 🔨 How to run the code?
Please follow the instructions:<br/>

1. HSFCS based channel selction:<br/>
```
python channel_selection.py --no_of_folds [K fold cross validation] --deap_dataset_path [deap dataset path] --datafiles_path [put the path where the subject wise .csv files are kept]
```
**Note:** Data Handling and Preprocessing step is same as subject dependent analysis. Here we merge .csv files of all subjects. 

2. Genetic Algorithm based feature selection:<br/>
```
python GA_feature_selection.py --subject [subject name] --datafiles_path [put the path where the subject wise .csv files are kept] --alpha [weight for classification accuracy] --numPop [size of the population in GA] <br/>--numGen [No of generations in GA --cxpb [crossover probability in GA] --mutpb [mutation probability in GA] --tournsize [tournsize in GA]
```

**Note:** Discreate Wavelet based feature extraction step is same as subject dependent analysis.
