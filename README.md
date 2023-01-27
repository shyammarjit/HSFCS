# A Hybrid Sequential Forward Channel Selection Method for Enhancing EEG-Based Emotion Recognition

[Shyam Marjit](shyammarjit.github.io), [Parag Jyoti Das](https://www.linkedin.com/in/paragjdas/), [Upasana Talukdar](https://www.iiitg.ac.in/faculty/upasana/), and [Shyamanta M Hazarika](https://www.iitg.ac.in/s.m.hazarika/)

[![paper](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/abstract/document/9588702)
[![code](https://img.shields.io/badge/code-80:20-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B80-20%5D.ipynb)
[![code](https://img.shields.io/badge/code-10--fold-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B10-fold%5D.ipynb)
[![result](https://img.shields.io/badge/result-80:20-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/80-20%20GA-MLP%20results.md)
[![result](https://img.shields.io/badge/result-10--fold-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/10-fold%20GA-MLP%20results.md)



## Structure of Code directory

```
├── Data files                                    <- Datafiles: stored features
│   ├── PSD features                              <- Power Spectral Density based features (.csv files)
│   ├── Wavelet features                          <- Discreate Wavelet based features (.csv files)
│
├── Subject Dependent                             <- Subject Dependent code
│   ├── Data Preprocessing                        <- Data Preprocessing [Band Pass Filtering -> ICA -> CAR]
│   ├── Channel Selection                         <- HSFCS based Channel Selection
│   ├── Wavelet based Features Extraction         <- Discreate Wavelet based feature extraction
│       ├── utils                                 <- Functions for features extraction
│   ├── Genetic Algorithm for Feature Selection   <- GA for feature selection from optimal channels
│       ├── utils_channels                        <- Subject wise optimal channels list
│
├── Subject Independent                           <- Subject Independent code
│   ├── Channel Selection                         <- Merged all subject wise data and
│   │                                                perform HSFCS based Channel Selection
│   ├── Genetic Algorithm for Feature Selection   <- GA for feature selection from optimal channels
```
