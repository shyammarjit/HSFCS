# A Hybrid Sequential Forward Channel Selection Method for Enhancing EEG-Based Emotion Recognition

[Shyam Marjit](shyammarjit.github.io), [Parag Jyoti Das](https://www.linkedin.com/in/paragjdas/), [Upasana Talukdar](https://www.iiitg.ac.in/faculty/upasana/), and [Shyamanta M Hazarika](https://www.iitg.ac.in/s.m.hazarika/)

[![paper](https://img.shields.io/badge/paper-Taylor%20%26%20Francis-green)](http://dx.doi.org/10.1080/0952813X.2023.2301367)
[![code](https://img.shields.io/badge/code-Sub--Dep-orange)](https://github.com/shyammarjit/HSFCS/tree/main/code/subject%20dependent)
[![code](https://img.shields.io/badge/code-Sub--Indep-orange)](https://github.com/shyammarjit/HSFCS/tree/main/code/subject%20independent)
[![datafiles](https://img.shields.io/badge/datafiles-psd-blue)](https://github.com/shyammarjit/HSFCS/tree/main/code/datafiles/psd)
[![datafiles](https://img.shields.io/badge/datafiles-wavelet-blue)](https://github.com/shyammarjit/HSFCS/tree/main/code/datafiles/wavelet)



## Structure of Code directory

```
├── Data files                                    <- Datafiles: stored features
│   ├── PSD features                              <- Power Spectral Density based features (.csv files)
│   ├── Wavelet features                          <- Discreate Wavelet based features (.csv files)
│
├── Subject Dependent                             <- Subject Dependent code
│   ├── Data Preprocessing                        <- Data Preprocessing [Band Pass Filtering -> ICA -> CAR]
│   ├── Channel Selection                         <- HSFCS based Channel Selection
│   ├── Wavelet based Features Extraction         <- Discreate Wavelet based features extraction
│       ├── utils                                 <- Functions for feature extraction
│   ├── Genetic Algorithm for Feature Selection   <- GA for feature selection from optimal channels
│       ├── utils_channels                        <- Subject wise optimal channels list
│
├── Subject Independent                           <- Subject Independent code
│   ├── Channel Selection                         <- Aggregate all subject wise data and
│   │                                                perform HSFCS-based Channel Selection
│   ├── Genetic Algorithm for Feature Selection   <- GA for feature selection from optimal channels
```

## ✏️ Citation
If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```bash
@article{shyam2024hsfcs,
    author = {Shyam Marjit, Parag Jyoti Das, Upasana Talukdar and Shyamanta M Hazarika},
    title = {A hybrid sequential forward channel selection method for enhancing EEG-Based emotion recognition},
    journal = {Journal of Experimental \& Theoretical Artificial Intelligence},
    volume = {0},
    number = {0},
    pages = {1-25},
    year = {2024},
    publisher = {Taylor & Francis},
    doi = {10.1080/0952813X.2023.2301367},
    URL = {https://doi.org/10.1080/0952813X.2023.2301367},
    eprint = {https://doi.org/10.1080/0952813X.2023.2301367},
}
```
