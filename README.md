# CoolSCA
The past years have witnessed a considerable increase in research efforts put into neural network-assisted profiled side-channel analysis (SCA). 
Studies have also identified challenges, e.g., closing the gap between metrics for machine learning (ML) classification and side-channel attack evaluation. One well-known fact has been overlooked in the SCA, namely NNs' tendency to become "over-confident," i.e., suffering from an overly high probability of correctness when predicting the correct class (secret key in the sense of SCA).
This repository reintroduces temperature scaling into SCA and demonstrates that key recovery can become more effective through that. 
Interestingly, temperature scaling can be easily integrated into SCA, and no re-tuning of the network is needed. 
In doing so, temperature can be seen as a metric to assess the NN's performance before launching the attack. 
In this regard, the impact of hyperparameter tuning, network variance, and capacity have been studied. 
This leads to recommendations on how network miscalibration and overconfidence can be prevented. 

# Dependencies
Install dependencies: tensorflow:
```bash
pip install tensorflow
```

scipy:
```python
pip install scipy
```

h5py:
```bash
pip install h5py
```

