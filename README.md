# Time-Series Attacks via STATistical Features
Python Implementation of Time-Series Attacks via STATistical Features (TSA-STAT) for the paper: "[Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features]()" by Taha Belkhouja, and Janardhan Rao Doppa.

## Setup 
```
pip install -r requirement.txt
```
By default, data is stored in `experim_path_{dataset_name}`. Directory can be changed in `RO_TS.py`


## Obtain datasets
- The dataset can be obtained as .zip file from "[The UCR Time Series Classification Repository](http://www.timeseriesclassification.com/dataset.php)", "[The UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/smartphone-based+recognition+of+human+activities+and+postural+transitions)", and "[The Activity Recognition using Cell Phone Accelerometers Repository](https://www.cis.fordham.edu/wisdm/dataset.php)".
- Download the .zip file and extract it it in `Dataset/{dataset_name}` directory.
- Run the following command for pre-processing a given dataset from The UCR Repository. For example, to extract SyntheticControl dataset
```
python preprocess_dataset.py --dataset_name=SyntheticControl 
```
The results will be stored in `Dataset` directory in [pickle](https://docs.python.org/3/library/pickle.html) format containing the training-testing examples with their corresponding labels.

## Run
- Example  training run
```
python train.py --dataset_name=SyntheticControl --window_size 61 --channel_dim 1 --class_nb 6
```

- Example TSA-STAT adversarial attack run
```
python tsa_stat_adversarial.py --dataset_name=SyntheticControl --window_size 61 --channel_dim 1 --class_nb 6 -t 0
```
The adversarial attack will be stored in [pickle](https://docs.python.org/3/library/pickle.html) file "TSASTAT_Attack.pkl"

- Example TSA-STAT certification run
```
python certify_model.py --dataset_name=SyntheticControl --window_size 61 --channel_dim 1 --class_nb 6 
```
The certification result  will be stored in [pickle](https://docs.python.org/3/library/pickle.html) file "Certificates.pkl"