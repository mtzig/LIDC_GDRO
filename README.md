# LIDC_GDRO
Improving Worst-Class Performance in Lung Nodule Malignancy Classification

Project Description:

Early detection of malignant lung nodules is vital to the treatment of lung cancer. Thus it is of great importance to develop effective techniques in Computer Aided Diagnosis (CAD) that can better help radiologists identify these lung nodules. With the recent advances in neural networks and deep learning, many novel deep learning models have been proposed to do this sort of classification. However while these models have good performance overall, they often fail in classification of critical subclasses of malignant nodules. Particularly, due to the heterogeneous mixture of features (e.g. spiculation, texture or sphericity) that decide the binary label of a lung nodule as malignant or not, malignant nodules can be divided into further subclasses of nodules grouped by these features. Thus in this project, our overall goal is to implement a deep learning model that has good worst-class performance without sacrificing overall performance. Specifically, we will attempt to do this by training our model on the Lung Image Database Consortium image collection (LIDC-IDRI) dataset using a Group Distributional Robust Optimization (GDRO) loss function in combination with certain lung nodule semantic features.

Code adapted from https://github.com/facebookresearch/DomainBed