# Detection of Malfunctions in the Operation of S-net Seafloor Observatories Using Machine Learning
## Annotation:
The goal of this project is to develop a model for automatically detecting faulty pressure sensors in the S-net seafloor observatory network. The main issue is that sometimes sensors malfunction and begin transmitting chaotic noise instead of actual ocean bottom pressure data. As a result, the accuracy of tsunami wave height predictions based on offshore pressure measurements decreases. By identifying and excluding faulty stations, we aim to solve this problem.

## Data:
This repository includes 14 days of pressure registration data from February 22 to March 8, 2025, for 256 pressure sensors in the `data` folder. The file `channels.tbl.20250303` contains a table with information about the sensors (name, coordinates, depth, etc.). The data is provided by the NIED organization: https://hinetwww11.bosai.go.jp/auth/oc/. The raw data is provided at a frequency of 10 Hz, which is excessive for our purposes. Therefore, the data fed into the model is resampled to a frequency of 1 point per hour by averaging.

If you use new data not included in the `data` folder, it must be resampled manually, which can be made with the `resampling.ipynb` file. The filenames of the raw data should match the sensor names (as listed in the `channels.tbl.20250303` file).

For training the model, the stations were labeled into two classes: `good` (functional) and `bad` (malfunctioning). A total of 29 stations were labeled as malfunctioning, while the remaining 227 were considered functional.

## Requirements:

The final.ipynb and other notebook files have been run with python 3.12.0 on Windows 10 OS with NVIDIA CUDA supported (Adapt all needed packages versions accroding your python version)

One part of the code requires the use of the TPXO regional tidal model for the Pacific Ocean. Due to the large file size, it cannot be hosted on GitHub. Therefore, please download the model in advance from the website (https://www.tpxo.net/regional, Pacific Ocean) and place the extracted folder `PO` with its contents into the repository.

The list of required libraries and their versions is provided in requirements.txt.

## Model execution guide:

The classification of the input stations into functional and malfunctioning is performed in two stages. In the first stage, the time series for each station is fed into six base classifiers: three different models combined with two loss functions. The output of this stage is a distribution of reconstruction errors for each station: the higher the error, the greater the likelihood that the station is malfunctioning. In the second stage, these error distributions are used to train a final meta-classifier.

All necessary code is located in the `final.ipynb` file.

### Base classifiers:
The code for the base classifiers is contained in the functions `method1`, ..., `method4`. Each base classifier takes the following inputs:

- `station_data` - a dictionary of time series for all stations
- `names` - an array that includes station information (names, coordinates, depths, etc.). This file is provided along with the pressure registration data when downloaded from the NIED website
- `test` - an array that includes the indices of stations to be used in the test set. All other stations are automatically used for training. By default, this array is empty

### Meta classifier:
The function `train_meta_classifier_with_plots` computes the final prediction for whether a station belongs to one of the two classes: functional (_good_) or malfunctioning (_bad_). The function takes the following inputs:

- `errors` – a list of 6 error arrays (3 methods, each with 2 error functions)
- `Testidx` – a list of indices for the stations to be used in the test set; the remaining stations are used for training
- `bad` – a list of indices of malfunctioning stations (labels). Stations not included in this list are considered functional
- `pics` – a boolean variable. If set to True, the function will display the ROC curve, feature importance plot, and a classification report. If set to False, these plots are not generated
- `method_names` – a list of method names to be displayed in the feature importance plot

The function returns the trained model, the normalizer, and the F1-score for the bad class, which is used as the main evaluation metric.

### Train-Test split:
The data may contain internal dependencies, so the train-test split is performed with certain mandatory conditions:

1) If two (or more) sensors are installed at the same station, they must either both go into the test set or both into the training set—never split between the two.
2) All time segments for the same station must be grouped together so that the classifier does not simply “memorize” station-specific patterns, but instead learns to distinguish between functional and malfunctioning stations.
3) The ratio of good to bad stations in the train and test sets should be approximately the same as in the original dataset. However, this ratio can be adjusted by changing the parameters K and N (see below).

The function takes the following inputs:

- `names` – an array used to identify which sensors belong to the same station
- `bad` – an array of indices of malfunctioning stations (the remaining are considered functional)
- `K` – the minimum number of malfunctioning (_bad_) sensors that must be included in the test set
- `N` – the minimum total number of stations that must be included in the test set
-  `seed` – a seed for random selecting stations according to the specified conditions

The function returns a list of sensor indices included in the test set. All remaining sensors are used for training.

### Model training:

In the last two cells, the final data processing and model training are performed. The original 14-day dataset is divided into 4 segments of 84 hours (3.5 days) each. Each segment is treated as a separate sample. All samples for each method are combined into one array of length 4N, where N is the number of sensors in the original dataset. Samples with indices k, k+N, k+2N, and k+3N correspond to the same station but over different time intervals.

The list of malfunctioning stations `worst` is similarly replicated 4 times, each with an offset of N, to account for the fact that if a station is malfunctioning, it remains so across all time segments.

The function `select_grouped_stations` is used to perform the train-test split.

If the variable `need_to_calculate` is set to `False`, error distributions are loaded from files in the `errors` folder. If it is set to `True`, all distributions are recalculated using the base classifiers (which is time-consuming). The classifier `method2` always recalculates the error distributions, as it is the only method whose results depend on the train/test split (which in turn depends on the random seed).

The error distributions for all methods are collected into a single list `errors`.

In the final cell, training of the meta-classifier is launched (which is relatively fast if the error distributions are already loaded or precomputed).

## Contacts:

To ask more questions about that project, leave any recomendations, suggestions and feedback about that project and its code be free to contact Oleg V. Ponomarev:
- Email (professional): ponomarev.ov20@physics.msu.ru
- Email (personal): bumerangxfox@gmail.com
- Telegram: https://t.me/devadevam

## Acknowledgments:

**Work is greatly supported by Non-commercial Foundation for the Advancement of Science and Education INTELLECT**

Also thanks a lot to https://github.com/Nikita-Belyakov for a huge assist for creating this project!
