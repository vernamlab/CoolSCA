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

## Acknowledgements

1. This project uses code from the [NN_calibration](https://github.com/markus93/NN_calibration) repository by Markus93 for temperature calibration.
2. The codes from [AutoSCA](https://github.com/AISyLab/AutoSCA/tree/main) were used to train the models.
3. We used the codes from [EnsembleSCA](https://github.com/AISyLab/EnsembleSCA) for the CNN/HW model for ASCADf dataset.
4. [TCHES20V3_CNN_SCAPublic](https://github.com/KULeuven-COSIC/TCHES20V3_CNN_SCA) and [Methodology-for-efficient-CNN-architectures-in-SCA](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA) were used for CNN/ID for ASCADf dataset.

