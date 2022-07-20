import os
import torch
import torchvision
import pandas as pd
import numpy as np
from train_eval import train, evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from matplotlib import pyplot as plt


def get_normed(this_array, this_min=0, this_max=255, set_to_int=True):
    """
        INPUTS:
        this_array: raw image from file

        OUTPUT:
        normalized version of image
    """

    rat = (this_max - this_min) / (this_array.max() - this_array.min())
    this_array = this_array * rat
    this_array -= this_array.min()
    this_array += this_min
    if set_to_int:
        return this_array.to(dtype=torch.int) / this_max
    return this_array / this_max


def scale_image(image_dim, upscale_amount=None, crop_change=None):
    """
        INPUTS:
        upscale_amount: amount to upscale image by, if None, upscales
                 to original size

        OUTPUTS:
        scalar: a function that returns the multichannel scaled version
                of a image
    """
    if not upscale_amount:
        upscale_amount = image_dim

    if not crop_change:
        crop_change = image_dim // 4

    crop_1_amount = image_dim
    crop_2_amount = image_dim - crop_change
    crop_3_amount = image_dim - 2 * crop_change

    upscale = torchvision.transforms.Resize(upscale_amount)
    crop_1 = torchvision.transforms.CenterCrop(crop_1_amount)
    crop_2 = torchvision.transforms.CenterCrop(crop_2_amount)
    crop_3 = torchvision.transforms.CenterCrop(crop_3_amount)

    def scalar(image):
        """
            INPUTS:
            Image: normalized image of shape (1, H, W)
            NOTE: H should equal W
            OUPUTS:
            scaled image: image with channels of different crops of
                          image, shape of (3, H, W)
        """

        img_ch1 = upscale(crop_1(image))
        img_ch2 = upscale(crop_2(image))
        img_ch3 = upscale(crop_3(image))
        image = torch.cat([img_ch1, img_ch2, img_ch3])

        return image

    return scalar


def get_malignancy(lidc_df, nodule_id, binary, device):
    malignancy = lidc_df[lidc_df['noduleID'] == nodule_id]['malignancy'].iloc[0]
    if binary:
        return torch.tensor(1, device=device) if malignancy > 1 else torch.tensor(0, device=device)

    return torch.tensor(malignancy, device=device) if malignancy > 1 else torch.tensor(malignancy,
                                                                                           device=device)


def get_subclass(lidc_df, nodule_id, sublabels):
    subtype = lidc_df[lidc_df['noduleID'] == nodule_id][sublabels].iloc[0]
    return subtype
    # if subtype == 'marked_benign':
    #     return torch.tensor(0, device=device)
    # elif subtype == 'unmarked_benign':
    #     return torch.tensor(1, device=device)
    # elif subtype == 'marked_malignant':
    #     return torch.tensor(2, device=device)
    # else:
    #     return torch.tensor(3, device=device)


def get_data_split(train_test_df, nodule_id):
    return train_test_df[train_test_df['noduleID'] == nodule_id]['split'].iloc[0]


def augment_image(image):
    """
        Input:
        image: tensor of shape (3, H, W)

        Ouput:
        tuple of image and its augmented versions
    """

    image_90 = torchvision.transforms.functional.rotate(image, 90)
    image_180 = torchvision.transforms.functional.rotate(image, 180)
    image_270 = torchvision.transforms.functional.rotate(image, 270)
    image_f = torch.flip(image, [0, 1])  # flip along x-axis

    return image, image_90, image_180, image_270, image_f


def get_images(image_folder='./data/LIDC(MaxSlices)_Nodules_Subgrouped',
               data_split_file='./data/train_test_splits/LIDC_data_split.csv',
               lidc_subgroup_file='./data/subclass_labels/LIDC_data_split_with_cluster.csv',
               image_dim=71,
               sublabels=None,
               split=True,
               binary=True,
               device='cpu'):
    """
        Input:
        image_folder: directory of the image files

        Output:
        m1: list of the labels encountered (1,2,4,5)
        m2: list of binary labels encountered (benign, malignant)
        diff: list of any nodes with discrepancy to CSV labels
    """

    train_img = []
    train_label = []
    train_subclasses = []

    cv_img = []
    cv_label = []
    cv_subclasses = []

    test_img = []
    test_label = []
    test_subclasses = []

    nodule_id = []

    lidc = pd.read_csv(lidc_subgroup_file)
    train_test = pd.read_csv(data_split_file)

    scalar = scale_image(image_dim)

    for dir1 in os.listdir(image_folder):

        if dir1 == 'Malignancy_3':
            continue

        for file in os.listdir(os.path.join(image_folder, dir1)):

            temp_nodule_id = int(file.split('.')[0])
            malignancy = get_malignancy(lidc, temp_nodule_id, binary, device)

            if sublabels:
                subtype = get_subclass(lidc, temp_nodule_id, sublabels)

            if split:
                split_type = get_data_split(train_test, temp_nodule_id)

            image_raw = np.loadtxt(os.path.join(image_folder, dir1, file))
            image_raw = torch.from_numpy(image_raw).to(device)
            image_normed = get_normed(image_raw).unsqueeze(dim=0)
            image = scalar(image_normed)

            if split and split_type == 0:
                images = augment_image(image)
                train_img.extend(images)
                train_label.extend([malignancy for _ in range(len(images))])
                if sublabels:
                    train_subclasses.extend([subtype for _ in range(len(images))])
            elif split and split_type == 1:
                cv_img.append(image)
                cv_label.append(malignancy)

                if sublabels:
                    cv_subclasses.append(subtype)
            else:
                test_img.append(image)
                test_label.append(malignancy)

                if sublabels:
                    test_subclasses.append(subtype)

                nodule_id.append(temp_nodule_id)

    if sublabels:
        train_data = (train_img, train_label, train_subclasses)
        cv_data = (cv_img, cv_label, cv_subclasses)
        test_data = (test_img, test_label, test_subclasses)
    else:
        train_data = (train_img, train_label)
        cv_data = (cv_img, cv_label)
        test_data = (test_img, test_label)

    if split:
        return train_data, cv_data, test_data
    else:
        return nodule_id, test_data


def get_cnn_features(feature_file='./data/erm_cluster_cnn_features_1.csv',
                     split_file='./data/subclass_labels/LIDC_data_split_with_cluster.csv', device='cpu', subclass='cluster'):
    df_features = pd.read_csv(feature_file, index_col=0)
    df_splits = pd.read_csv(split_file, index_col=0)
    df = df_features.sort_values('noduleID')
    df['clusters'] = df_splits[subclass]
    df['malignancy_b'] = df_splits['malignancy_b']

    dfs = []
    for i in range(3):
        dfs.append(df[df_splits['split'] == i])

    datas = []
    for d in dfs:
        X = torch.tensor(d.drop(['noduleID', 'clusters', 'malignancy_b'], axis=1).values,
                         device=device, dtype=torch.float32)
        y = torch.tensor(d['malignancy_b'].values, device=device)
        c = torch.tensor(d['clusters'].values, device=device)
        datas.append((X, y, c))

    return datas


def get_train_val_split(dataset, split_percent=0.8):
    train_size = int(split_percent * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, (train_size, val_size))


def train_epochs(epochs, train_loader, val_loader, model, loss_fn, scheduler=True, num_subgroups=None,
                 verbose=True):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.005)
    if scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train(train_loader, model, loss_fn, optimizer, verbose=verbose)
        accuracies = evaluate(val_loader, model, num_subgroups, verbose=verbose)

        if scheduler:
            scheduler.step(accuracies[0])


def show_scatter(component_0, component_1, group, title, size):
    fig, ax = plt.subplots()
    group = group
    component_0 = component_0
    component_1 = component_1
    # legend = {0:'red', 1:'blue'}

    for g in np.unique(group):
        idx = np.where(group == g)
        ax.scatter(component_0[idx], component_1[idx], label=g, s=size)  # c = legend[g],

    ax.legend()
    plt.title(title)
    plt.show()
