import os
import random
import tqdm
import numpy as np
from scipy.stats import sem
import scipy.stats as stats
from pytorchtools import EarlyStopping
import yaml
import wandb

from SegmentationDataset import SegmentationDataset

import torch
import torch.utils.data as data_utils
import torchvision


def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.backbone.conv1 = torch.nn.Conv2d(5,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.aux_classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))
    model.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1, 1))

    return model

def load_dataset(whole_dataset_path, batch_size):

    train_stream = []
    test_stream = []
    dataset_name_list = []
    for root, folders, files in os.walk(whole_dataset_path):
        if root != whole_dataset_path:
            break

        for folder in folders:
            dataset_name_list.append(folder)
            print(os.path.join(root,folder))
            dataset = SegmentationDataset(os.path.join(root,folder))
            num_of_samples = len(dataset)
            index_training = random.sample(range(1,num_of_samples), int(0.9*num_of_samples))

            train_dataset = data_utils.Subset(dataset, index_training)
            train_dataloader = data_utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

            test_dataset = data_utils.Subset(dataset, list(set(range(1,num_of_samples)).difference(set(index_training))))
            test_dataloader = data_utils.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=6, pin_memory = True, prefetch_factor=batch_size*2)

            train_stream.append(train_dataloader)
            test_stream.append(test_dataloader)

    print("Datasets name", dataset_name_list)

    return train_stream, test_stream

def train_model(model:torch.nn.Module, loss_function, optimizer, device, epoch_num, epoch, train_datasetloader:data_utils.DataLoader):
    model.train()
    model.to(device=device)

    sum_loss = 0
    step_num = len(train_datasetloader)

    with tqdm.tqdm(total= step_num) as tbar:
        for data, target in train_datasetloader:
            data, target = data.to(device), target.to(device)
            if data.shape[0] == 1:
                continue
            output = model(data)["out"]
            # print(output.shape, target.shape)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss
            
            tbar.set_description('Training Epoch: {}/{} Loss: {:.6f}'.format(epoch, epoch_num, loss.item()))
            tbar.update(1)
    
    ave_loss = sum_loss / step_num
    return ave_loss


def val_model(model:torch.nn.Module, device, loss_function, val_datasetloader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = 0
    with torch.no_grad():
        with tqdm.tqdm(total = len(val_datasetloader)) as pbar:
            for data, target in val_datasetloader:
                data, target = data.to(device), target.to(device)
                output = model(data)["out"]
                loss = loss_function(output, target)
                pred = torch.argmax(output, 1)
                correct += torch.sum(pred == target)
                total_num += pred.shape.numel()
                print_loss = loss.data.item()
                test_loss += print_loss
                pbar.update(1)

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(val_datasetloader)
        # print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     avgloss, correct, total_num, 100 * acc))
    
    return avgloss, correct, acc

def compute_acc_fgt(end_task_acc_arr):
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1 + 0.95) / 2, n_run - 1)  # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]  # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)  # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), np.std(avg_acc_per_run),t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), np.std(avg_fgt),t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), np.std(acc_per_run), t_coef * sem(acc_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc


def load_parameter_from_yaml(yaml_path):
    with open(yaml_path,'r', encoding='utf-8') as parameter_file:
        args = yaml.load(parameter_file, Loader=yaml.FullLoader)

    print("Load parameters: ", args)
    return args


if __name__ == "__main__":

    args = load_parameter_from_yaml("experiments.yaml")
    data_path = args["data_path"]
    run_times = args["run_times"]
    batch_size = args["batch_size"]
    learn_rate = args["learn_rate"]
    basic_task_index = args["basic_task_index"]
    epoch_num = args["epoch_num"]
    patience = args["patience"]
    pretrained_model_path = args["pretrained_model_path"]
    basic_model_save_path = args["basic_model_save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(
        project="lifelong-hyperspecial",
        config=args
    )

    # record for multiple run
    accuracy_list1 = [] 
    accuracy_list2 = []
    accuracy_list3 = []
    accuracy_list4 = []
    fun_score = np.zeros((run_times, 4))

    for time in range(run_times):
        print("Start {} times experiences".format(time+1))
        model = load_model()
        model.to(device)

        train_stream, test_stream = load_dataset(data_path, batch_size)

        loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
        optimizer = torch.optim.AdamW(model.parameters(),learn_rate)

        # training basic task
        train_dataloader = train_stream[basic_task_index]
        test_dataloader = test_stream[basic_task_index]

        acc_array1 = np.zeros((4, 2))
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=basic_model_save_path)

        # model.load_state_dict(torch.load(pretrained_model_path))
        for epoch in range(epoch_num):
            training_loss = train_model(model, loss_function, optimizer, device, epoch_num,epoch, train_dataloader)
            avgloss, _, segment_acc = val_model(model, device, loss_function, test_dataloader)
            early_stopping(avgloss, segment_acc, model, training_loss)
            log = {f'training loss init model_{time}': training_loss, 
                f'val loss init model_{time}': avgloss, 
                f'val acc init model_{time}': segment_acc, 
                'epoch':epoch}
            wandb.log(log)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        basic_loss = early_stopping.training_loss
        print("basic loss:{:.4}".format(basic_loss))
        acc_array1[:,0] = early_stopping.acc

        # pop the src data from train_stream and test_stream
        train_stream.pop(basic_task_index)
        test_stream.pop(basic_task_index)

        for i, probe_data in enumerate(test_stream):
            with torch.no_grad():
                _, _, acc_array1[i,1] = val_model(model, device, loss_function, probe_data)

        del model
        del train_dataloader

        torch.cuda.empty_cache()

        acc_array2 = np.zeros((4, 2))
        for j, probe_data in enumerate(train_stream):
            print("task {} starting...".format(j))

            trained_model = load_model()
            trained_model.load_state_dict(torch.load(basic_model_save_path))

            optimizer = torch.optim.AdamW(trained_model.parameters(),learn_rate)

            trained_model.to(device)

            for k, one_batch in enumerate(probe_data):
                x_train, y_train = one_batch[0].to(device), one_batch[1].to(device)
                y_pred = trained_model(x_train)["out"]
                loss = loss_function(y_pred, y_train)
                # calculate the functional similarity (fun_score) at last steps
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if k==0:
                    print("loss value: ", loss.data.item())
                    fun_score[time, j] = (1 - (loss.data.item() / basic_loss)) 
                    break

            early_stopping = EarlyStopping(patience=patience, verbose=True)
            for epoch in range(epoch_num):
                train_model(trained_model, loss_function,optimizer, device, epoch_num, epoch, probe_data)
                avgloss, _, segment_acc = val_model(trained_model, device, loss_function, test_stream[j])
                early_stopping(avgloss, segment_acc, trained_model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            trained_model.load_state_dict(torch.load("checkpoint.pt"))

            _, _, acc_array2[j,0] = val_model(trained_model, device, loss_function, test_dataloader)
            _, _, acc_array2[j,1] = val_model(trained_model, device, loss_function, test_stream[j])
        
        
        accuracy_list1.append([acc_array1[0, :], acc_array2[0, :]])
        accuracy_list2.append([acc_array1[1, :], acc_array2[1, :]])
        accuracy_list3.append([acc_array1[2, :], acc_array2[2, :]])
        accuracy_list4.append([acc_array1[3, :], acc_array2[3, :]])

        with open("log.txt",'a') as log_file_each:
            print(f"===========start num times {time}============================")
            print(f"func_score: ", fun_score[time, :])
            print(f"basic model acc: ", acc_array1[:,0])
            print(f"average acc: ", (acc_array2[:,0]+acc_array2[:,1])/2 )
            print(f"fgt: ", acc_array1[:,0]-acc_array2[:,0])
            print("Original data", acc_array1, acc_array2)
            print(f"===========end num times {time}==============================")
            print(f"===========start num times {time}============================", file=log_file_each)
            print(f"func_score: ", fun_score[time, :], file=log_file_each)
            print(f"basic model acc: ", acc_array1[:,0], file=log_file_each)
            print(f"average acc: ", (acc_array2[:,0]+acc_array2[:,1])/2 , file=log_file_each)
            print(f"fgt: ", acc_array1[:,0]-acc_array2[:,0], file=log_file_each)
            print("Original data", acc_array1, acc_array2, file=log_file_each)
            print(f"===========end num times {time}==============================", file=log_file_each)
        del trained_model

    with open("log/log.txt",'wt') as log_file:
        fun_score_mean = np.mean(fun_score, axis=0)
        fun_score_std = np.std(fun_score, axis=0)
        print(f"fun_score_mean {fun_score_mean}")
        print(f"fun_score_std {fun_score_std}")
        print(f"fun_score_mean {fun_score_mean}", file=log_file)
        print(f"fun_score_std {fun_score_std}", file=log_file)

        accuracy_array1 = np.array(accuracy_list1)
        accuracy_array2 = np.array(accuracy_list2)
        accuracy_array3 = np.array(accuracy_list3)
        accuracy_array4 = np.array(accuracy_list4)

        avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array1)
        print('--Task 1------ Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
        print('--Task 1--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc), file=log_file)
        avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array2)
        print('--Task 2--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
        print('--Task 2--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc), file=log_file)
        avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array3)
        print('--Task 3--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
        print('--Task 3--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc), file=log_file)
        avg_end_acc, avg_end_fgt, avg_acc = compute_acc_fgt(accuracy_array4)
        print('--Task 4--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc))
        print('--Task 4--- Avg_End_Acc {} Avg_End_Fgt {} Avg_Acc {}-----------'.format(avg_end_acc, avg_end_fgt, avg_acc), file=log_file)
