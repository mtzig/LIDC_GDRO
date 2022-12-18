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
                 model,
                 loss_fn,
                 optimizer,
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
        accuracies_BEST = []

    max_val_accuracy = 0
    max_worst_accuracy = 0
    best_epoch = 0

    best_model = None

    for epoch in range(epochs):
        if verbose:
            print(f'Epoch {epoch + 1} / {epochs}')

        train(train_dataloader, model, loss_fn, optimizer)

        temp_val_accuracies = evaluate(val_dataloader, model, num_subclasses=num_subclasses)
        temp_val_accuracy = temp_val_accuracies[0]
        temp_worst_accuracy = min(temp_val_accuracies[1:])
        
        if verbose:
            print('val_overall: ', temp_val_accuracy)
            print('val_worst: ', temp_worst_accuracy)
        

        if erm_flag:           
            if temp_val_accuracy >= max_val_accuracy:
                max_val_accuracy = temp_val_accuracy
                best_epoch = epoch

                if verbose:
                    print('I am saving the best ERM model at Epoch: ',best_epoch)

                best_model = model.state_dict().copy()
                best_epoch = epoch

        else:
            if temp_worst_accuracy >= max_worst_accuracy:
                max_worst_accuracy = temp_worst_accuracy
                best_epoch = epoch

                if verbose:
                    print('I am saving the best gdro model at Epoch: ', best_epoch)

                best_model = model.state_dict().copy()
                best_epoch = epoch



        if scheduler and erm_flag:
            scheduler.step(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[0])

            if verbose:
                print('ERM Learning learning rate is: ', optimizer.param_groups[0]['lr'])
        
        if scheduler and not erm_flag:
            scheduler.step(min(evaluate(val_dataloader, model, num_subclasses=num_subclasses)[1:]))

            if verbose:
                print('DRO Learning learning rate is: ', optimizer.param_groups[0]['lr'])

        if record:
            epoch_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses)
            accuracies.extend(epoch_accuracies)
    


    

    if record:
        model.load_state_dict(best_model)
        best_test_accuracies = evaluate(test_dataloader, model, num_subclasses=num_subclasses)
        accuracies_BEST.extend(best_test_accuracies)

        return accuracies, accuracies_BEST, best_epoch
    else:
        return None


def init_scheduler(scheduler_config, optimizer):  
    
        scheduler_class = getattr(schedulers, scheduler_config['class_name'])
        return scheduler_class(optimizer, **scheduler_config['class_args'])
    


def run_trials(num_trials,
               epochs,
               train_dataloader,
               val_dataloader,
               test_dataloader,
               model_class,
               model_args,
               loss_class,
               loss_args,
               optimizer_class,
               optimizer_args,
               device='cpu',
               num_subclasses=1,
               scheduler=None,
               verbose=False,
               record=False
               ):
    if record:
        accuracies = []
        accuracies_best = []
        best_epochs = []


    for n in range(num_trials):
        if verbose:
            print(f"Trial {n + 1}/{num_trials}")

        model = model_class(*model_args).to(device)
        loss_args[0] = model
        loss_fn = loss_class(*loss_args)
        optimizer = optimizer_class(model.parameters(), **optimizer_args)

        trial_results = train_epochs(epochs,
                                     loss_class is not GDROLoss,
                                     train_dataloader,
                                     val_dataloader,
                                     test_dataloader,
                                     model,
                                     loss_fn,
                                     optimizer,
                                     scheduler=scheduler,
                                     verbose=verbose,
                                     record=record,
                                     num_subclasses=num_subclasses)

        if record:
            trial_accuracies, accuracies_BEST, best_epoch = trial_results
            accuracies.extend(trial_accuracies)
            accuracies_best.extend(accuracies_BEST)
            best_epochs.append(best_epoch)



    if record:
        return accuracies, accuracies_best, best_epochs
    else:
        return None
