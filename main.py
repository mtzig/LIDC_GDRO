from torch.utils.data import DataLoader

import GDRO
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

from datasets import NoduleDataset


def main():
    # import data
    df = pd.read_csv("LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv")
    subtype_df = pd.read_csv("lidc_subtyped.csv")

    # select features and labels
    id_name = 'noduleID'
    feature_names = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter',
                     'MajorAxisLength', 'MinorAxisLength', 'SuperscribedDiameter',
                     'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent',
                     'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness',
                     'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG',
                     'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG',
                     'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4',
                     'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1',
                     'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0',
                     'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2',
                     'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1',
                     'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0',
                     'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2',
                     'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity',
                     'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage',
                     'Variance', 'Clustertendency']
    label_name = 'malignancy'
    df = df.loc[:, [id_name, *feature_names, label_name]]

    # remove malignancy = 3
    df = df[df[label_name] != 3]

    # binarize the remaining malignancy [1,2] -> 0, [4,5] -> 1
    df[label_name] = [int(m - 3 > 0) for m in df[label_name]]

    # normalize numeric features
    df.loc[:, feature_names] = StandardScaler().fit_transform(df.loc[:, feature_names].values)

    # separate into training and test sets
    training_fracion = 0.8

    training_df = df.sample(frac=training_fracion)
    test_df = df.drop(training_df.index)

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create the training and testing dataloaders
    batch_size = 4

    training_data = torch.FloatTensor(training_df.loc[:, feature_names].values).to(device)
    training_labels = torch.LongTensor(training_df.loc[:, label_name].values).to(device)
    test_data = torch.FloatTensor(test_df.loc[:, feature_names].values).to(device)
    test_labels = torch.LongTensor(test_df.loc[:, label_name].values).to(device)

    train_dataloader = DataLoader(NoduleDataset(training_data, training_labels), batch_size=batch_size)
    test_dataloader = DataLoader(NoduleDataset(test_data, test_labels), batch_size=batch_size)

    # create and train model
    model = GDRO.NeuralNetwork(64, 32, 32, 2)

    loss_fn = GDRO.ERMLoss(model, torch.nn.CrossEntropyLoss(), {})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        GDRO.train(train_dataloader, model, loss_fn, optimizer)
        GDRO.test(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    main()
