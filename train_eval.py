import numpy as np

import torch
from loss import GDROLoss

from utils import data_utils, image_data_utils
from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset
import torch.optim.lr_scheduler as schedulers


def train(dataloader, model, loss_fn, optimizer, verbose=False):
    """
    Train the model for one epoch
    :param dataloader: The dataloader for the training data
    :param model: The model to train
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param verbose: Whether to print the average training loss of the epoch
    :return:
    """
    model.train()

    steps_per_epoch = dataloader.batches_per_epoch()

    avg_loss = 0

    for i in range(steps_per_epoch):
        loss = loss_fn(next(dataloader))
        avg_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss /= steps_per_epoch

    if verbose:
        print("Average training loss:", avg_loss)


def evaluate(dataloader, model, num_subclasses, verbose=False):
    """
    Evaluate the model's accuracy and subclass sensitivities
    :param dataloader: The dataloader for the validation/testing data
    :param model: The model to evaluate
    :param num_subclasses: The number of subclasses to evaluate on, this should be equal to the number of subclasses present in the data
    :param verbose: Whether to print the results
    :return: A tuple containing the overall accuracy and the sensitivity for each subclass
    """
    model.eval()

    num_samples = np.zeros(num_subclasses)
    subgroup_correct = np.zeros(num_subclasses)
    with torch.no_grad():
        X = dataloader.dataset.features
        y = dataloader.dataset.labels
        c = dataloader.dataset.subclasses

        pred = model(X)

        for subclass in range(num_subclasses):
            subclass_idx = c == subclass
            num_samples[subclass] += torch.sum(subclass_idx)
            subgroup_correct[subclass] += (pred[subclass_idx].argmax(1) == y[subclass_idx]).type(
                torch.float).sum().item()

    subgroup_accuracy = subgroup_correct / num_samples

    accuracy = sum(subgroup_correct) / sum(num_samples)

    if verbose:
        print("Accuracy:", accuracy, "\nAccuracy over subgroups:", subgroup_accuracy, "\nWorst Group Accuracy:",
              min(subgroup_accuracy))

    return (accuracy, *subgroup_accuracy)


def train_epochs(epochs,
                 erm_flag,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 eval_train_dataloader,
                 model,
                 loss_fn,
                 optimizer,
                 device,
                 scheduler=None,
                 verbose=False,
                 record=False,
                 num_subclasses=4
                 ):
    """
    Trains the model for a number of epochs and evaluates the model at each epoch
    :param epochs: The number of epochs to train
    :param train_dataloader: The dataloader for the training data
    :param test_dataloader: The dataloader for the validation/testing data
    :param model: The model to train and evaluate
    :param loss_fn: The loss function to use for training
    :param optimizer: The optimizer to use for training
    :param scheduler: The learning rate scheduler, if any, to use for training
    :param verbose: Whether to print the epoch number and the results for each epoch
    :param record: Whether to return the results from evaluate, generally this should be True
    :param num_subclasses: The number of subclasses to evaluate on
    :return: A list containing the overall accuracy and subclass sensitivities for each epoch, arranged 1-dimensionally ex. [accuracy_1, subclass1_1, subclass2_1, accuracy_2, subclass1_2, subclass2_2...]
    """
    if record:
        accuracies = []
        accuracies_train = []
        accuracies_validation = []
        accuracies_BEST = []
        max_val_accuacy = 0
        max_worst_accuracy = 0
        best_epoch = 0
        # q_data = None
        # if isinstance(loss_fn, GDROLoss):
        #     q_data = []

    best_model = None

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        train(train_dataloader, model, loss_fn, optimizer)

        temp_val_accuracies = evaluate(val_dataloader, model, num_subclasses=num_subclasses)
        temp_val_accuracy = temp_val_accuracies[0]
        temp_worst_accuracy = min(temp_val_accuracies[1:])
        
        print('val_overall: ', temp_val_accuracy)
        print('val_worst: ', temp_worst_accuracy)
        

        if erm_flag:           
            if temp_val_accuracy >= max_val_accuacy:
                max_val_accuacy = temp_val_accuracy
                best_epoch = epoch
                print('I am saving the best ERM model at Epoch: ',best_epoch)
                # torch.save(model.state_dict(), 'D:\LIDC_GDRO_UseValidation\Best_model.pth')
                best_model = model.state_dict().copy()

        else:
            if temp_worst_accuracy >= max_worst_accuracy:
                max_worst_accuracy = temp_worst_accuracy
                best_epoch = epoch
                print('I am saving the best dro model at Epoch: ', best_epoch)
                # torch.save(model.state_dict(), 'D:\LIDC_GDRO_UseValidation\Best_model.pth')
                best_model = model.state_dict().copy()


        if scheduler and erm_flag:
            scheduler.step(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[0])
            print('ERM Learning learning rate is: ', optimizer.param_groups[0]['lr'])
        
        if scheduler and not erm_flag:
            scheduler.step(min(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[1:]))
            print('DRO Learning learning rate is: ', optimizer.param_groups[0]['lr'])

        if record:
            epoch_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses)
            epoch_accuracies_train = evaluate(eval_train_dataloader, model, num_subclasses=num_subclasses)
            epoch_accuracies_val = evaluate(val_dataloader, model, num_subclasses=num_subclasses)
            accuracies.extend(epoch_accuracies)
            accuracies_train.extend(epoch_accuracies_train)
            accuracies_validation.extend(epoch_accuracies_val)
    
    # Loading
    

    # model.load_state_dict(torch.load('D:\LIDC_GDRO_UseValidation\Best_model.pth'))
    model.load_state_dict(best_model)
    model = model.to(device)
    best_test_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses)
    accuracies_BEST.extend(best_test_accuracies)

           

    if record:
        return accuracies, accuracies_train, accuracies_validation,accuracies_BEST
    else:
        return None




def generate_dataloaders_designedFeatures (df, individual_split_path, fn_name,batch_size):
    
    train, val, test = data_utils.train_val_test_datasets(df, split_path=individual_split_path)
    
    val_dataloader = InfiniteDataLoader(SubclassedDataset(*val), len(val))
    test_dataloader = InfiniteDataLoader(SubclassedDataset(*test), len(test))
    eval_train_dataloader = InfiniteDataLoader(SubclassedDataset(*train), len(train))
    num_subclasses = len(test_dataloader.dataset.subclasses.unique())
    
    tr = SubclassedDataset(*train)
    
    if fn_name == 'ERMLoss':
        train_dataloader = InfiniteDataLoader(tr, batch_size)
    else:
        train_dataloader = InfiniteDataLoader(tr, batch_size, weights=image_data_utils.get_sampler_weights(tr.subclasses))
    
    return train_dataloader, val_dataloader, test_dataloader, num_subclasses, eval_train_dataloader






def run_trials_designedFeatures(
               epochs,
               df,
               csv_files,
               fn_name,
               batch_size,
            #    train_dataloader,
            #    test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args_erm,
               optimizer_args_dro,
               device='cpu',
               num_subclasses=1,
               scheduler=None,
               verbose=False,
               record=False
               ):
    if record:
        accuracies = []
        accuracies_train = []
        accuracies_val = []
        accuracies_test_BEST = []
        # roc_data = [None, None]
        # q_data = None
        # g_data = None
        # if loss_class is GDROLoss:
        #     q_data = []

    num_trials = len(csv_files)
    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        individual_split_path = csv_files[n]
        train_dataloader, val_dataloader, test_dataloader, num_subclasses,eval_train_dataloader = generate_dataloaders_designedFeatures(df, individual_split_path, fn_name,batch_size)        
        
        model = model_class(*model_args).to(device)
        loss_args[0] = model
        loss_fn = loss_class(*loss_args)
        

        if fn_name == 'ERMLoss':
            erm_flag = True
            optimizer = optimizer_class(model.parameters(), **optimizer_args_erm)
        else:
            erm_flag = False
            optimizer = optimizer_class(model.parameters(), **optimizer_args_dro)
        
        scheduler = init_scheduler({'class_args': {'patience':10,'factor': 0.1,'mode':'min'},'class_name': 'ReduceLROnPlateau'},optimizer)



        trial_results = train_epochs(epochs,
                                     erm_flag,
                                     train_dataloader,
                                     val_dataloader,
                                     test_dataloader,
                                     eval_train_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     device,
                                     scheduler,
                                     verbose,
                                     record,
                                     num_subclasses
                                    )

        if record:
            trial_accuracies, trial_accuracies_train,trial_accuracies_val,best_test_accuracies = trial_results
            accuracies.extend(trial_accuracies)
            accuracies_train.extend(trial_accuracies_train)
            accuracies_val.extend(trial_accuracies_val)
            accuracies_test_BEST.extend(best_test_accuracies)

    if record:
        return accuracies, accuracies_train, accuracies_val,accuracies_test_BEST
    else:
        return None

def init_scheduler(scheduler_config, optimizer):  
    
        scheduler_class = getattr(schedulers, scheduler_config['class_name'])
        return scheduler_class(optimizer, **scheduler_config['class_args'])
    



def run_trials_images(
               epochs,
               csv_files,
               fn_name,
               batch_size,
               subclass_column,
            #    train_dataloader,
            #    test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args_erm,
               optimizer_args_dro,
               device='cpu',
             
            #    scheduler=True,
               verbose=False,
               record=False
               ):
    if record:
        accuracies = []
        accuracies_train = []
        accuracies_val = []
        accuracies_test_BEST = []
        # roc_data = [None, None]
        # q_data = None
        # g_data = None
        # if loss_class is GDROLoss:
        #     q_data = []

    
    
    # num_trials = len(csv_files)
    # For Thomas Experiment
    num_trials = 10

    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        # individual_split_path = csv_files[n]
        individual_split_path = csv_files[2] # for thomas experiment
        train, val, test = image_data_utils.get_features(split_file = individual_split_path,device=device, images=True, subclass=subclass_column)

        val_dataloader = InfiniteDataLoader(SubclassedDataset(*val), len(val))
        test_dataloader = InfiniteDataLoader(SubclassedDataset(*test), len(test))
        eval_train_dataloader = InfiniteDataLoader(SubclassedDataset(*train), len(train))

        num_subclasses = len(test_dataloader.dataset.subclasses.unique())

        tr = SubclassedDataset(*train)

        if fn_name == 'ERMLoss':
            train_dataloader = InfiniteDataLoader(tr, batch_size)

        else:
            train_dataloader = InfiniteDataLoader(tr, batch_size, weights=image_data_utils.get_sampler_weights(tr.subclasses))


        
        model = model_class(*model_args).to(device)
        loss_args[0] = model
        loss_fn = loss_class(*loss_args)

        if fn_name == 'ERMLoss':
            optimizer = optimizer_class(model.parameters(), **optimizer_args_erm)
        elif fn_name == 'GDROLoss':
            optimizer = optimizer_class(model.parameters(), **optimizer_args_dro)
        else:
            print('ERROR!! IN run_trials_images')

        scheduler = init_scheduler({'class_args': {'patience':10,'factor': 0.1,'mode':'min'},'class_name': 'ReduceLROnPlateau'},optimizer)


        if fn_name == 'ERMLoss':
            erm_flag = True
        else:
            erm_flag = False

        trial_results = train_epochs(epochs,
                                     erm_flag,
                                     train_dataloader,
                                     val_dataloader,
                                     test_dataloader,
                                     eval_train_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     device,
                                     scheduler,
                                     verbose,
                                     record,
                                     num_subclasses
                                    )

        if record:
            trial_accuracies, trial_accuracies_train,trial_accuracies_val,best_test_accuracies = trial_results
            accuracies.extend(trial_accuracies)
            accuracies_train.extend(trial_accuracies_train)
            accuracies_val.extend(trial_accuracies_val)
            accuracies_test_BEST.extend(best_test_accuracies)

    #         if isinstance(loss_fn, GDROLoss):
    #             q_data.extend(trial_q_data)


    #         with torch.no_grad():
    #             preds = model(test_dataloader.dataset.features)
    #             probabilities = torch.nn.functional.softmax(preds, dim=1)[:, 1]
    #             if roc_data[0] is None:
    #                 roc_data[0] = probabilities
    #             else:
    #                 roc_data[0] += probabilities
    # if record:
    #     roc_data[0] /= num_trials
    #     labels = test_dataloader.dataset.labels
    #     roc_data[1] = labels

    if record:
        return accuracies, accuracies_train, accuracies_val,accuracies_test_BEST
    else:
        return None




def run_trials_CNN(
               epochs,
               csv_files,
               fn_name,
               batch_size,
               subclass_column,
            #    train_dataloader,
            #    test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args,
               device='cpu',
             
               scheduler=None,
               verbose=False,
               record=False
               ):
    if record:
        accuracies = []
        roc_data = [None, None]
        q_data = None
        g_data = None
        if loss_class is GDROLoss:
            q_data = []

    
    
    num_trials = len(csv_files)

    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        individual_split_path = csv_files[n]
        train, val, test = image_data_utils.get_features(split_file = individual_split_path,device=device,subclass=subclass_column)

        val_dataloader = InfiniteDataLoader(SubclassedDataset(*val), len(val))
        test_dataloader = InfiniteDataLoader(SubclassedDataset(*test), len(test))

        num_subclasses = len(test_dataloader.dataset.subclasses.unique())

        tr = SubclassedDataset(*train)

        if fn_name == 'ERMLoss':
            train_dataloader = InfiniteDataLoader(tr, batch_size)
        else:
            train_dataloader = InfiniteDataLoader(tr, batch_size, weights=image_data_utils.get_sampler_weights(tr.subclasses))


        
        model = model_class(*model_args).to(device)
        loss_args[0] = model
        loss_fn = loss_class(*loss_args)
        optimizer = optimizer_class(model.parameters(), **optimizer_args)

        trial_results = train_epochs(epochs,
                                     train_dataloader,
                                    #  test_dataloader,
                                     val_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     scheduler,
                                     verbose,
                                     record,
                                     num_subclasses)

        if record:
            trial_accuracies, trial_q_data = trial_results
            accuracies.extend(trial_accuracies)

            if isinstance(loss_fn, GDROLoss):
                q_data.extend(trial_q_data)


            with torch.no_grad():
                preds = model(test_dataloader.dataset.features)
                probabilities = torch.nn.functional.softmax(preds, dim=1)[:, 1]
                if roc_data[0] is None:
                    roc_data[0] = probabilities
                else:
                    roc_data[0] += probabilities
    if record:
        roc_data[0] /= num_trials
        labels = test_dataloader.dataset.labels
        roc_data[1] = labels

    if record:
        return accuracies, q_data, roc_data
    else:
        return None