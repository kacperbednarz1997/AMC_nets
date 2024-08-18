# AMC_nets

Kacper Bednarz

# Models
Code containing various models for automatic modulation classification (AMC).
The models were implemented in the PyTorch library and were taken from the literature:
1) CNN [1]

![](/assets/CNN_architektura.png)
2) RNN [2]

![](/assets/RNN_architektura.png)
3) LSTM [2]

![](/assets/LSTM_architektura.png)
4) CLDNN [1]

![](/assets/CLDNN_architektura.png)
5) CGDNN [1]

![](/assets/CGDNN_architektura.png)
6) ResNet [3]

![](/assets/ResNet_architektura.png)
7) AWN [39][40]

![](/assets/AWN_architektura.png)

# Preparation
## Datasets
The experiments were performed on four datasets: RML2016.10a, RML2016.10b, RML2018.01a and MIGOU-MOD. The download links as of the date of the repository upload are in the table below.

| dataset     | modulations                                         | link to download         |
| ----------- | ------------------------------------------------------------ | ------------------------ |
| RML2016.10a | 8PSK, BPSK, CPFSK, GFSK, PAM4, 16QAM, 64QAM, QPSK, AM-DSB，AM-SSB，WBFM |[[RML2016.10a](https://www.kaggle.com/datasets/raindrops12/rml201610a)]|
| RML2016.10b | 8PSK, BPSK, CPFSK, GFSK, PAM4, 16QAM, 64QAM, QPSK,AM-DSB，WBFM |[[RML2016.10b](https://www.kaggle.com/datasets/marwanabudeeb/rml201610b)]|
| RML2018.01a | 32PSK, 16APSK, 32QAM, GMSK, 32APSK, OQPSK, 8ASK, BPSK, 8PSK, 4ASK, 16PSK, 64APSK, 128QAM, 128APSK, 64QAM, QPSK, 256QAM, OOK, 16QAM, AM-DSB-WC, AM-SSB-WC, AM-SSB-SC, AM-DSB-SC, FM, |[[RML2018.01a](https://www.kaggle.com/datasets/pinxau1000/radioml2018)]|
| MIGOU-MOD | 8PSK, BPSK, CPFSK, GFSK, PAM4, 16QAM, 64QAM, QPSK, AM-DSB，AM-SSB，WBFM |[[MIGOU-MOD](https://data.mendeley.com/datasets/fkwr8mzndr/1)]|

Please extract the downloaded compressed file to the ./datasets folder and keep the file name unchanged. The ./dataset directory is located parallel to the ./AMC_nets directory. The final directory structure should be shown below:：

```
├─ AMC_nets
├─ datasets
    ├── AMC_nets
    ├── GOLD_XYZ_OSC.0001_1024.hdf5
    ├── RML2016.10a_dict.pkl
    ├── RML2016.10b.dat
    └── migou_dataset_19.08_400000x128.pkl
```

## Pretrained models
Pretrained models are located in the ./checkpoints folder. 
There is no checkpoint for the LSTM model due to the file size being too large.

## Environment Setup
- Python == 3.12.3
- PyTorch == 2.4.0 

These are the versions on which the scripts were run.

# Training and evaluation

In the ./AMC_nets folder there is a script "main.py".

Changing the "models" variable selects which model is currently to be trained/evaluated.
The "datasets" variable tells about the available datasets.
To switch the mode from training to evaluation, change the default "mode" variable.

After appropriate changes and running the "main.py" script, the program should be started.

After each training, a single evaluation is started automatically. The training results, including: logs, model and inference results are located in the ./AMC_nets/training folder.

If you want to run a single evaluation without training, you must first move the trained model with the .pkl extension (sample path: /AMC_nets/training/2016.10a_0/model/2016.10a_CNN_Pijackova.pkl) to the ./AMC_nets/checkpoint folder.

# Acknowledgments
Some of the code is borrowed from [[AWN](https://github.com/zjwfufu/AWN?tab=readme-ov-file)]. I sincerely thank them for their outstanding work.

# License
This code is distributed under an [MIT LICENSE](https://github.com/kacperbednarz1997/AMC_nets/blob/main/LICENSE). Note that my code depends on other libraries and datasets which each have their own respective licenses that must also be followed.

# Literature:
[1] K. Pijáčková, „Radio modulation recognition networks”, Bachelor’s Thesis, Brno University of Technology, Brno, 2021

[2] E. Salama i N. Hesham, Modulation Classification. University of Alexandria, 2023

[3] T. O’Shea, R. Tamoghna, i C. T. Charles, „Over the Air Deep Learning Based Radio Signal Classification”, 2017

[4] J. Zhang, T. Wang, Z. Feng, i S. Yang, „Toward the Automatic Modulation Classification With Adaptive Wavelet Network”, IEEE Transactions on Cognitive Communications and Networking, t. 9, nr 3, s. 549–563, cze. 2023, doi: 10.1109/TCCN.2023.3252580
