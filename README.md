# NHNN Implementation

This is the Pytorch implementation for paper "Accounting for Variations in Speech Emotion Recognition with NonParametric Hierarchical Neural Network".

The training file included is for IEMOCAP. The metadata is stored in data.csv, which includes audio segment id, subject id, gender label, and emotion label (valence rating). The eGeMAPS features for the audios are extracted and stored as features.csv. 

The Log-MFB features can be downloaded [here](https://drive.google.com/file/d/1b0c7xgcyzt97vY7AI5uOCHdYWhTfygFf/view?usp=sharing)

To run the training for CNN, you can simply run

```
python3 train_CNN.py
```

To run the training for NHNN, type

```
python3 train_NHNN.py *version*
```

Here, version must be either FC or FC+Conv

Please note that the current NHNN implementation requires a feature encoder. Therefore you must run the training for CNN first, and the feature encoder will be used for training the NHNN model. 
