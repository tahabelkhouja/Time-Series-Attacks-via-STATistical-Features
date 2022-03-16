# Time-Series Attacks via STATistical Features
Python Implementation of Time-Series Attacks via STATistical Features (TSA-STAT) for the paper: "[Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features]()" by Taha Belkhouja, and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirement.txt
```
By default, data is stored in `experim_path_{dataset_name}`. Directory can be changed in `RO_TS.py`


## Obtain datasets
- The dataset can be obtained as .zip file from "[The UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php)".
- Download the .zip file and extract it it in `UCRDatasets/{dataset_name}` directory.
- Run the following command for pre-processing a given dataset while specifying if it is multivariate, for example, on SyntheticControl dataset
```
python preprocess_dataset.py --dataset_name=SyntheticControl --multivariate=False
```
The results will be stored in `Dataset` directory. 

## Run
- Example  training run
```
python tsa_stat.py --dataset_name=SyntheticControl 
```

- Example testing run
```
python test_tsa_stat.py --dataset_name=SyntheticControl 
```

