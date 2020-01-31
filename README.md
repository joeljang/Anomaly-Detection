# Anomaly-Detection
Novel anomaly detection method using a data wrangling process and convolution neural networks

Calculate_ae_md.py : Changing raw data into a health-indicator value using autoencoders and mahalanobis distance.
Calculate_rms.py : Chainging raw vibration data into root-mean-squared(RMS) Health indicator value
Creatensp.py : a novel data wrangling method, Nested-Scatter-Plot, that transforms raw (vibration data) into an image with time-frequency analysis so that the data can be changed from time-series data into an image that can be used as input for Convolution Neural Networks
Featuremap.py : Using a CNN-autoencoder to extract features from the NSP image and using tSNE and PCA to do dimension reduction for feature map analysis.
Fft.py : Using Fast-Fourier transform to designate the bandpass filter frequencies before using creatnsp.py
Traincnn.py: Using NSP images to train CNN-Binary Regression model for early anomlay detection
