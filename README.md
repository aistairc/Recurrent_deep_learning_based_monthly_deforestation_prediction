# Recurrent Deep Learning-based Monthly Deforestation Prediction

This scripts are described in the paper, A Recurrent Deep Learning-based Monthly Deforestation Prediction Model in Eight Areas for Brazilian Amazon: A Pilot Study, submitted to IGARSS 2024.<br />

### \<Overview\>
- Python scripts in this repository create $1\times1$ km meshes of eight areas (Altamira, Humaita, Novo Progresso, Porto Velho, S6W57, S7W57, Sao Felix do Xingu, and Vista Alegre do Abuna) from preprocessed about seven-year data (August 2016 to September 2023) using the Real-Time Deforestation Detection System (DETER), construct recurrent deep learning-based prediction model consisting of two long short term memory (LSTM) layers and two fully-connected layers with a softmax output layer, and evaluate the mesh-wise model performance by applying an incremental traininig and testing method for two-year data (October 2021 to September 2023).<br />
  - __features__: the area where deforestation occurred and the binary of whether deforestation occurred or not<br />
  - __input sequence length__: 12 months<br />
  
- 
