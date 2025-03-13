import gc
import math
import os
import random
import sys
import time
import torch
import dataLoader
from datetime import datetime
from torch import nn
from torch.optim import AdamW
from torchmetrics.classification import BinaryAccuracy
from EEGNet import EEGNetModel as EEGNet
# from models.ATCNet import ATCNet_ as ATCNet
import matplotlib.pyplot as plt

def train(model_name, train_loader, val_loader, learning_rate, epochs, batch_size, output_path:str, seed=None, validate=True, model_path=None, val_loss = 0):
    """ Function to train the model.
        Params:
          - model: the model to be trained
          - train_data: training data (Pandas DataFrame format)
          - val_data: validation data (Pandas DataFrame format)
          - learning_rate: learning rate
          - epochs: the number of epochs for training
    """
    start_time = datetime.now()
    version = start_time.strftime("%d.%m_%H.%M")
    if not validate:
        save_path = os.path.join(output_path, "models", version + "_retrain")
        log_path = os.path.join(output_path, "logs", version + "_retrain")
    else:
        save_path = os.path.join(output_path, "models", version)
        log_path = os.path.join(output_path, "logs", version)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    summary_log = open(os.path.join(log_path, "#summary.txt"), 'w')
    summary_log.write(f"batch_size: {batch_size} \nepochs: {epochs}")

    valid_result = []
    train_result = []
    train_result_acc = []
    val_result_acc = []

    nchannels = train_loader.dataset.data[0].shape[0]

    print(f"Get model {model_name}")
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    if model_name == 'EEGNet':
        model = EEGNet(chans=nchannels, classes=1, time_points=250, temp_kernel=32,
                       f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.5)
    else:
        raise ValueError("Model not found")

    if not validate:
        state_dict = torch.load(model_path, weights_only=False)
        model.load_state_dict(state_dict, strict=False)

    print("Building optimizer")
    criterion = nn.BCEWithLogitsLoss()

    acc = BinaryAccuracy(threshold=0.5)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        acc = acc.cuda()
        print("Using Cuda")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Training")
    length = len(train_loader)
    lowest_val_loss = math.inf
    counter = 0
    for epoch_num in range(1, epochs + 1):
        model.train()

        total_loss_train = 0

        time_begin = time.time()
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M:%S")
        print(f"epoch: {epoch_num} time: {current_time}")
        i = 0
        # epoch_log = open(os.path.join(log_path, "epoch_log " + str(epoch_num) + ".txt"), 'w')

        for train_input, train_label in train_loader:
            i += 1

            train_label = train_label.float().unsqueeze(1).to(device)
            train_input = train_input.float().unsqueeze(1).to(device)

            # model.zero_grad()
            optimizer.zero_grad()

            output = model(train_input)

            # _, predicted = torch.max(output.data, 1)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc(output.data, train_label)

            batch_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            sys.stdout.flush()
            gc.collect()

            # step_time = time.time()
            # elapsed_time = step_time - time_begin
            # batch_time = elapsed_time / i
            # remaining_time = batch_time * (length - i)
            #
            # remaining_hours = remaining_time // 3600
            # remaining_minutes = floor(((remaining_time / 3600) - remaining_hours) * 60)
            # remaining_seconds = floor(((remaining_time / 60) - floor(remaining_time / 60)) * 60)
            #
            # elapsed_hours = elapsed_time // 3600
            # elapsed_minutes = floor(((elapsed_time / 3600) - elapsed_hours) * 60)
            # elapsed_seconds = floor(((elapsed_time / 60) - floor(elapsed_time / 60)) * 60)

            # log = f"[{epoch_num}/{epochs}]: [{i}/{length}] loss:{batch_loss:.5f} " \
            #       f"lr:{learning_rate:.6f} elapsed time: {elapsed_hours:.0f}:{elapsed_minutes:.0f}:{elapsed_seconds:.0f} " \
            #       f"time remaining: {remaining_hours:.0f}:{remaining_minutes:.0f}:{remaining_seconds:.0f}"

            # epoch_log.write(log + "\n")
            # if i % 10 == 0:
            #     print(log)

            # if i >= 2:  # debug
            #     break

        train_acc = acc.compute()
        acc.reset()

        total_loss_val = 0

        if validate:
            print("Validating")
            model.eval()
            i = 0

            with torch.no_grad():
                for val_input, val_label in val_loader:
                    i += 1

                    val_label = val_label.float().unsqueeze(1).to(device)
                    if model_name == 'FBCNet':
                        val_input = val_input.float().to(device)
                    else:
                        val_input = val_input.float().unsqueeze(1).to(device)

                    if model_name == 'Conformer':
                        _, output = model(val_input)
                    else:
                        output = model(val_input)

                    # _, predicted = torch.max(output.data, 1)

                    val_loss = criterion(output, val_label)
                    total_loss_val += val_loss.item()

                    acc(output.data, val_label)

                    sys.stdout.flush()
                    gc.collect()

                    # if i >= 2:  # debug
                    #     break

            learning_rate = optimizer.param_groups[0]["lr"]

            val_acc = acc.compute()
            acc.reset()

            avg_val_loss = total_loss_val / len(val_loader)
            avg_train_loss = total_loss_train / len(train_loader)

            train_log = f"EPOCH {epoch_num} TRAIN avloss: {avg_train_loss:.6f} Acc: {train_acc:.6f}"
            val_log = f"EPOCH {epoch_num} VALID avloss: {avg_val_loss:.6f} Acc5: {val_acc:.6f}"

            # epoch_log.write(train_log + "\n")
            # epoch_log.write(val_log)
            # epoch_log.close()

            summary_log.write(f"{train_log}\t")
            summary_log.write(f"{val_log}\n")

            print(train_log)
            print(val_log)

            train_result.append(avg_train_loss)
            valid_result.append(avg_val_loss)

            train_result_acc.append(train_acc.cpu())
            val_result_acc.append(val_acc.cpu())

            model_path = os.path.join(save_path, f"{model_name}_{version}_epoch_{epoch_num}.pt")

            torch.save(model.state_dict(), model_path)
            model.load_state_dict(torch.load(model_path, weights_only=False))

            if avg_val_loss > lowest_val_loss:
                counter += 1
                if counter == 30:
                    epochs = epoch_num
                    print("Early stop")
                    break
            else:
                lowest_val_loss = avg_val_loss
                counter = 0
        else:
            avg_train_loss = total_loss_train / len(train_loader)
            train_log = f"EPOCH {epoch_num} TRAIN avloss: {avg_train_loss:.6f} Acc: {train_acc:.6f}"
            summary_log.write(f"{train_log}\t")

            print(train_log)

            train_result.append(avg_train_loss)
            train_result_acc.append(train_acc.cpu())

            model_path = os.path.join(save_path, f"{model_name}_{version}_epoch_{epoch_num}.pt")

            torch.save(model.state_dict(), model_path)
            model.load_state_dict(torch.load(model_path, weights_only=False))

            if avg_train_loss <= val_loss:
                epochs = epoch_num
                print("Early stop")
                break

    del model
    torch.cuda.empty_cache()

    summary_log.close()

    return train_result, valid_result, train_result_acc, val_result_acc, version, epochs


if __name__ == "__main__":
    path = os.path.join("..", "Datasets", "PhysioNet")

    LR = 2e-4
    EPOCHS = 300
    batch_size = 24
    subject = '1'
    model_name = 'FACTNet'  #['EEGNet', 'FACTNet']

    train_loader, val_loader, _, _ = dataLoader.load(path, subject, batch_size, use_filter=True)
    output_path = "output"

    train_result, valid_result, train_result_acc, val_result_acc, version, epochs = train(model_name, train_loader, val_loader, LR, EPOCHS, batch_size, output_path)

    log_path = os.path.join(output_path, "logs", model_name, version)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, epochs+1), train_result)
    ax1.plot(range(1, epochs+1), valid_result)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend(['Train', 'Validation'])
    fig1.savefig(os.path.join(log_path, "loss.png"))

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, epochs+1), train_result_acc)
    ax2.plot(range(1, epochs+1), val_result_acc)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Train vs Validation Accuracy')
    ax2.legend(['Train', 'Validation'])
    fig2.savefig(os.path.join(log_path, "accuracy.png"))

    fig1.show()
    fig2.show()

