

# Robust Anomaly Detection via Extra-Long Attention Mechanism for Multivariate Time Series

In the process of anomaly detection, monitoring and analysis of multivariate time series are very crucial to the quality of service in the industry. Multivariate time series will generally contain complex time characteristics of the device and change of its status, it is effective to learn normal modes wherein the time series from which to distinguish anomaly. This paper proposes a method to split the original data and use random concatenation to generate time subsequences to learn more complex data distributions. Then we designed a mask method of a random continuous segment or random discrete points in a certain range of position and a corresponding objective function for masked data and overall sequence. These methods are applied to the dual-stream self-attention model to learn a good distribution of training data in normal mode. Anomaly detection uses the reconstruction error between the reconstructed data output by the model and the original test data. The evaluation experiments are conducted on three public datasets from aerospace and a public server machine dataset from an Internet company. In this paper, our model has a total F1 score of 0.9492 in three real datasets, which is significantly better than the best performing baseline method.


## Directory

- [File Directory](#FileDirectory)
- [Framework](#Framework)
- [Author](#Author)

###### ****

1. Get a free API Key 
2. Clone the repo

```sh
git clone https://github.com/Phil-Shawn/ELAD.git
```

### FileDirectory
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






