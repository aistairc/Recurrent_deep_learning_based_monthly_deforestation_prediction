# Recurrent Deep Learning-based Monthly Deforestation Prediction

This scripts are described in the paper, A Recurrent Deep Learning-based Monthly Deforestation Prediction Model in Eight Areas for Brazilian Amazon: A Pilot Study, accepted by IGARSS 2024.<br />

### \<Overview\>
- Python scripts in this repository create $1\times1$ km meshes of eight areas (Altamira, Humaita, Novo Progresso, Porto Velho, S6W57, S7W57, Sao Felix do Xingu, and Vista Alegre do Abuna) from preprocessed about seven-year data (August 2016 to September 2023) using the Real-Time Deforestation Detection System (DETER), construct recurrent deep learning-based prediction model consisting of two long short term memory (LSTM) layers and two fully-connected layers with a softmax output layer, and evaluate the mesh-wise model performance by applying an incremental traininig and testing method for two-year data (October 2021 to September 2023).<br />
  - __features__: the area where deforestation occurred and the binary of whether deforestation occurred or not<br />
  - __input sequence length__: 12 months<br />
  
- The deforestation prediction performances of the recurrent deep learning-based models for $1\times1$ km meshes with 1-month resolution averaged over eight areas in Brazil Amazon are summarized in <a href="https://github.com/aistairc/Reccurent_deep_learning_based_monthly_deforestation_prediction/blob/main/model_performance.jpg?raw=true" target="_blank">model_performance.jpg</a>.
- A comparative analysis of actual and predicted deforestation events spanning a four-month interval, from June to September, in both 2022 and 2023 at Vista Alegre do Abuna is shown in <a href="https://github.com/aistairc/Reccurent_deep_learning_based_monthly_deforestation_prediction/blob/main/comparative_analysis.jpg?raw=true" target="_blank">comparative_analysis.jpg</a>.

### \<Scripts\>
- main.py contains all processing steps.<br />

### \<Environments\>
- checked OS: Windows 10 and Ubuntu 20.04
- Anaconda 3
- Python 3.6.13
- PyTorch 1.10.2 + cuda 11.3
- scikit-learn 0.24.2
- skorch 0.11.0
- The virtual environment configuration is provided as environment.yml
  - You can reproduce the environment by<br/>
```conda env create -f environment.yml```<br/>
