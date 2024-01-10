# CoolSCA
The past years have witnessed a considerable increase in research efforts put into neural network-assisted profiled side-channel analysis (SCA). 
Studies have also identified challenges, e.g., closing the gap between metrics for machine learning (ML) classification and side-channel attack evaluation. 
In fact, in the context of NN-assisted SCA, the NN's output distribution forms the basis for successful key recovery. 
In this respect, related work has covered various aspects of integrating neural networks (NNs) into SCA, including applying a diverse set of NN models, model selection and training, hyperparameter tuning, etc. 
Nevertheless, one well-known fact has been overlooked in the SCA-related literature, namely NNs' tendency to become "over-confident," i.e., suffering from an overly high probability of correctness when predicting the correct class (secret key in the sense of SCA).  
Temperature scaling is among the powerful and effective techniques that have been devised as a remedy for this. 
Regarding the principles of deep learning, it is known that temperature scaling does not affect the NN's accuracy; however, its impact on metrics for secret key recovery, mainly guessing entropy, is worth investigating. 
This paper reintroduces temperature scaling into SCA and demonstrates that key recovery can become more effective through that. 
Interestingly, temperature scaling can be easily integrated into SCA, and no re-tuning of the network is needed. 
In doing so, temperature can be seen as a metric to assess the NN's performance before launching the attack. 
In this regard, the impact of hyperparameter tuning, network variance, and capacity have been studied. 
This leads to recommendations on how network miscalibration and overconfidence can be prevented. 

# Dependencies
Install dependencies on Ubuntu: tensorflow:
'''pip install tensorflow
