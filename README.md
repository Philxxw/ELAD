

# Robust Anomaly Detection via Extra-Long Attention Mechanism for Multivariate Time Series

We proposes  a  model  to  segment  the  original  data  and  generate time  subsequences. And we use  the  random  concatenation  within  a certain time range to learn more complex data distributions. Then we  designed  a  mask  method  of  random  continuous  segments  or random  discrete  points  in  partial  position  range  of  data  and  a corresponding objective function for this method. These methods are  applied  to  the  dual-stream  self-attention  model  to  learn  a good  distribution  of  training  data.  Anomaly  detection  depend on  construction  error  between  the  reconstructed  data  output  by the model and the original test data. The evaluation experiments are conducted on two public datasets from aerospace and a public server  machine  dataset  from  an  Internet  company.  Our  model has a total F1 score of 0.9526 in the three real datasets, which is significantly  better  than  the  best  performing  baseline.


## Directory

- [File Directory](#File Directory)
- [Framework](#Framework)
- [Author](#Author)

###### ****

1. Get a free API Key 
2. Clone the repo

```sh
git clone https://github.com/Phil-Shawn/Robust-Anomaly-Detection-via-Extra-Long-Attention-Mechanism-for-Multivariate-Time-Series.git
```

### File Directory
```
Anomaly Detection
├── README.md
├── data_preprocess.py#Raw data is processed into subsequences
├── data_utils_train.py#Generate input data with Random Concatenation and Mask ids
├── data_utils_test.py
├── model.py#Attention model
├── main.py#Train
├── test.py#Test
├── eval_methods.py
├── eval_.py#Anomaly Detection
├── /prepar_input_MSL/
│  │  ├── /model_par/
│  │  ├── /train_data/
│  │  └── /test_data/
├── /prepar_input_SMAP/
│  │  ├── /model_par/
│  │  ├── /train_data/
│  │  └── /test_data/
├── /prepar_input_SMD/
│  │  ├── /model_par/
│  │  ├── /train_data/
│  │  └── /test_data/
├── /ServerMachineDataset/#Original Data
│  │  ├── /interpretation_label/
│  │  ├── /train/
│  │  ├── /test/
│  │  └── /test_label/
└──  /SMAP_and_MSL/#Original Data
│  │  ├── /train/
│  │  ├── /test/
      └── /test_label/
```

### Framework

- [Pytorch 1.3.0 ]
- [Python 3.6.8]

### Author

byrxxw@bupt.edu.cn






