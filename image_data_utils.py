import os
import torch
import torchvision
import pandas as pd
import numpy as np

from dataloaders import SubtypedDataLoader, InfiniteDataLoader

def getNormed(this_array, this_min = 0, this_max = 255, set_to_int = True):
    '''
        INPUTS:
        this_array: raw image from file

        OUTPUT:
        normalized version of image
    '''

    
    rat = (this_max - this_min)/(this_array.max() - this_array.min())
    this_array = this_array * rat
    this_array -= this_array.min()
    this_array += this_min
    if set_to_int:
        return this_array.to(dtype= torch.int) / this_max
    return this_array / this_max

def scaleImage(image_dim, upscale_amount = None, crop_change=None):
    '''
        INPUTS:
        upscale_amount: amount to upscale image by, if None, upscales
                 to original size

        OUTPUTS:
        scalar: a function that returns the multichannel scaled version
                of a image

    '''
    if not upscale_amount:
        upscale_amount = image_dim

    if not crop_change:
        crop_change = image_dim // 4

    crop_1_amount = image_dim
    crop_2_amount = image_dim - crop_change
    crop_3_amount = image_dim - 2*crop_change

    upscale = torchvision.transforms.Resize(upscale_amount)
    crop_1 = torchvision.transforms.CenterCrop(crop_1_amount)
    crop_2 = torchvision.transforms.CenterCrop(crop_2_amount)
    crop_3 = torchvision.transforms.CenterCrop(crop_3_amount)

    def scalar(image):
        '''
            INPUTS:
            Image: normalized image of shape (1, H, W)
            NOTE: H should equal W
            OUPUTS:
            scaled image: image with channels of different crops of
                          image, shape of (3, H, W)

        '''
        
        img_ch1 = upscale(crop_1(image))
        img_ch2 = upscale(crop_2(image))
        img_ch3 = upscale(crop_3(image))
        image = torch.cat([img_ch1,img_ch2,img_ch3])

        return image

    return scalar

def get_malignancy(lidc_df, nodule_id, device):

    malignancy = lidc_df[lidc_df['noduleID']==nodule_id]['malignancy'].iloc[0]
    return torch.tensor(1, device=device) if malignancy > 3 else torch.tensor(0, device=device)

def get_subtype(lidc_df, nodule_id, device):

    subtype = lidc_df[lidc_df['noduleID']==nodule_id]['subgroup'].iloc[0]
    
    if subtype == 'marked_benign':
        return torch.tensor(0, device=device)
    elif subtype == 'unmarked_benign':
        return torch.tensor(1, device=device)
    elif subtype == 'marked_malignant':
        return torch.tensor(2, device=device)
    else:
        return torch.tensor(3, device=device)

def get_data_split(train_test_df, nodule_id, device):      

    return train_test_df[train_test_df['noduleID'] ==nodule_id]['dataset'].iloc[0]

def augmentImage(image):
    '''
        Input:
        image: tensor of shape (3, H, W)

        Ouput:
        tuple of image and its augmented versions

    '''

    image_90 = torchvision.transforms.functional.rotate(image, 90)
    image_180 = torchvision.transforms.functional.rotate(image, 180)
    image_270 = torchvision.transforms.functional.rotate(image, 270)
    image_f = torch.flip(image, [0,1]) #flip along x-axis

    return image, image_90, image_180, image_270, image_f

def getImages(image_folder='./LIDC(MaxSlices)_Nodules(fixed)', 
              data_split_file = './data/lidc_train_test_split_stratified.csv', 
              lidc_subgroup_file='./data/lidc_spic_subgrouped.csv',
              image_dim = 71,
              split = True,
              device='cpu'):
    '''
        Input:
        image_folder: directory of the image files

        Output:
        m1: list of the labels encountered (1,2,4,5)
        m2: list of binary labels encountered (benign, malignant)
        diff: list of any nodes with discrepency to CSV labels

    '''
    train_img = []
    train_label = []
    train_subclasses = []

    test_img = []
    test_label = []
    test_subclasses = []

    lidc = pd.read_csv(lidc_subgroup_file)
    train_test = pd.read_csv(data_split_file)
    
    scalar = scaleImage(image_dim)

    for dir1 in os.listdir(image_folder):
  
        if dir1 == 'Malignancy_3':
            continue

        for file in os.listdir(os.path.join(image_folder, dir1)):

            temp_nodule_ID = int(file.split('.')[0])
            malignancy = get_malignancy(lidc, temp_nodule_ID, device)
            subtype = get_subtype(lidc, temp_nodule_ID, device)


            split_type = get_data_split(train_test, temp_nodule_ID, device)
            
            
            image_raw = np.loadtxt(os.path.join(image_folder, dir1,file))
            image_raw = torch.from_numpy(image_raw).to(device)
            image_normed = getNormed(image_raw).unsqueeze(dim=0)
            image = scalar(image_normed)

            if split_type == 'train' and split:
                images = augmentImage(image)
                train_img.extend(images)
                train_label.extend([malignancy for _ in range(len(images))])
                train_subclasses.extend([subtype for _ in range(len(images))])
            else: 
                test_img.append(image)
                test_label.append(malignancy)
                test_subclasses.append(subtype)


    train_data = (train_img, train_label, train_subclasses)
    test_data = (test_img, test_label, test_subclasses)


    return train_data, test_data if split else test_data


def getTrainValSplit(dataset, split_percent=0.8):

    train_size = int(split_percent * len(dataset))
    val_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, (train_size,val_size))

def getSubtypedDataLoader(dataset, batch_size, num_classes=4):
    subtype_data = []

    #inefficient way to get data from dataset
    loader = InfiniteDataLoader(dataset, batch_size=len(dataset))

    X, y, c = next(loader)
    for subclass in range(num_classes):
        subclass_idx = subclass == c

        features = torch.unbind(X[subclass_idx])
        label = y[subclass_idx][0]

        subtype_data.append((features, label))


    return SubtypedDataLoader(subtype_data, batch_size, singular=True)
