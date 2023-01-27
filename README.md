# A Hybrid Sequential Forward Channel Selection Method for Enhancing EEG-Based Emotion Recognition

[Shyam Marjit](shyammarjit.github.io), [Parag Jyoti Das](https://www.linkedin.com/in/paragjdas/), [Upasana Talukdar](https://www.iiitg.ac.in/faculty/upasana/), and [Shyamanta M Hazarika](https://www.iitg.ac.in/s.m.hazarika/)

[![paper](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/abstract/document/9588702)
[![code](https://img.shields.io/badge/code-80:20-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B80-20%5D.ipynb)
[![code](https://img.shields.io/badge/code-10--fold-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B10-fold%5D.ipynb)
[![result](https://img.shields.io/badge/result-80:20-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/80-20%20GA-MLP%20results.md)
[![result](https://img.shields.io/badge/result-10--fold-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/10-fold%20GA-MLP%20results.md)



## Structure of Code directory

```
├── **Subject Dependent**   <- List of developers and maintainers.
|     ├── [Data Preprocessing](https://github.com/shyammarjit/HSFCS/blob/main/code/subject%20dependent/data_preprocessing.py)
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
```
## :<br/>
1. [Data Handling, PreProcessing [Bandpass (4-48 Hz) + ICA + CAR]](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Dependent/Code/STEP-1.%20%5Bband:%204-48%5D%20%5Bfir%5D%20%5Bno_of_components%20%3D%2031%5D%20%5Bone%20component%20dropping%5D.ipynb)<br/>
2. [HSFCS-based Channel Selection](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Dependent/Code/STEP-2.%20HSFCS%20based%20Channel%20Selection%20BCI.ipynb)<br/>
3. [Wavelet-based Feature Extraction]() [\[datafiles\]](https://github.com/shyammarjit/HSFCS/tree/main/Subject%20Independent/data%20files/Wavelet%20Based)<br/>
4. [Multi-objective Genetic Algorithm based Feature Selection]()<br/>
&emsp;&emsp;&emsp;&emsp; [High/Low Valence](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Dependent/Code/STEP-3.A.%20%5BValence%5D-GA%20based%20feature%20selection.ipynb)<br/>
&emsp;&emsp;&emsp;&emsp; [High/Low Arousal](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Dependent/Code/STEP-3.C.%20%5BArousal%5D-GA%20based%20feature%20selection.ipynb)<br/>
&emsp;&emsp;&emsp;&emsp; [HVHA/HVLA/LVLA/LVHA](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Dependent/Code/STEP-4.A.%20%5BFour-class%5D-GA%20based%20feature%20selection.ipynb)<br/>


## Subject Independent:**<br/>
1. [All subjects preprocessed datafiles](https://github.com/shyammarjit/HSFCS/tree/main/Subject%20Dependent/band_48_fir_None_one) \[same as subject dependent\]<br/>
2. [Channel Selection](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Independent/Code/Step-1.%20Subject%20Independent%20channel%20selection%20BCI-Copy2.ipynb)<br/>
3. [Wavelet-based Feature Extraction]()<br/>
4. [Multi-objective Genetic Algorithm based Feature Selection]()<br/>
&emsp;&emsp;&emsp;&emsp; [High/Low Valence](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Independent/Code/STEP-2.A.%20Valence-GA%20based%20feature%20selection.ipynb)<br/>
&emsp;&emsp;&emsp;&emsp; [High/Low Arousal](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Independent/Code/STEP-2.b.%20Arousal-GA%20based%20feature%20selection.ipynb)<br/>
&emsp;&emsp;&emsp;&emsp; [HVHA/HVLA/LVLA/LVHA](https://github.com/shyammarjit/HSFCS/blob/main/Subject%20Independent/Code/STEP-2.b.%20Four%20class-GA%20based%20feature%20selection.ipynb)<br/>

