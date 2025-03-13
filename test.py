import gc
import os
import sys

import pandas as pd
from torch import nn

import dataLoader
import torch
# import pandas as pd
from torchmetrics.classification import BinaryAccuracy
from EEGNet import EEGNetModel as EEGNet



def test(model_name, test_loader, model_path):
    # current_model = model_options[model_name][0]
    # hidden_layer = model_options[model_name][1]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    nchannels = test_loader.dataset.data[0].shape[0]

    print(f"Load model {model_path[-1][:-3]}")
    if model_name == 'EEGNet':
        model = EEGNet(chans=nchannels, classes=1, time_points=250, temp_kernel=32,
                       f1=16, f2=32, d=2, pk1=8, pk2=16, dropout_rate=0.5, max_norm1=1, max_norm2=0.25)
    else:
        raise ValueError("Model not found")

    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Retrieving data")

    torch.cuda.empty_cache()

    acc = BinaryAccuracy(threshold=0.5)

    criterion = nn.BCEWithLogitsLoss()

    if use_cuda:
        acc = acc.cuda()
        criterion = criterion.cuda()


    full_output = []
    total_loss_test = 0
    total_diff = 0

    i = 0
    print("Start testing")
    for test_input, test_label in test_loader:
        i += 1
        test_label = test_label.float().unsqueeze(1).to(device)
        test_input = test_input.float().unsqueeze(1).to(device)

        output = model(test_input)

        full_output.extend(output.detach().cpu().numpy())
        # _, predicted = torch.max(output.data, 1)

        total_diff += torch.mean(torch.abs(torch.sub(output.data, test_label))).item()

        acc(output.data, test_label)

        batch_loss = criterion(output, test_label)
        total_loss_test += batch_loss.item()

        torch.cuda.empty_cache()
        sys.stdout.flush()
        gc.collect()

        # if i >= 5:  # debug
        #     break

    test_acc = acc.compute()
    acc.reset()

    avg_test_loss = total_loss_test / i
    avg_diff = total_diff / i

    # output_folder = os.path.join(model_path, "..", "classification")
    output_folder = os.path.join(os.path.dirname(model_path), "classification")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_data = pd.DataFrame({"classification": full_output})
    output_data.to_csv(os.path.join(output_folder, "classification.csv"), index=False, lineterminator="\r\n")

    print(
        f"Loss:{avg_test_loss:.3f} acc:{test_acc:.3f} diff:{avg_diff:.3f}")

    return test_acc.cpu().numpy(), avg_test_loss, avg_diff


if __name__ == "__main__":
    modelname = "EEGNet"
    # version = "20.11_10.34"
    # epoch = 15
    # model_path = f"models/{modelname}/{version}/{modelname}_{version}_epoch_{epoch}.pt"

    path = os.path.join("..", "Datasets", "PhysioNet")
    subject = 1
    batch_size = 24

    half_sparse = ['Fc5', 'Fc1', 'C3', 'Cz', 'Cp5', 'Cp1', 'Fp1', 'Af3', 'F7', 'F3', 'Fz', 'T7', 'P7', 'P3', 'Pz', 'Po3', 'O1', 'Oz']

    # _, _, test_loader, _ = dataLoader.load(path, subject, batch_size, use_filter=True, channel_set=half_sparse)

    # model_path = "output/models/FACTNet/FACTNet_20.02_14.39_epoch_7.pt"
    model_path =  "output/models/EEGNet/mimm.pt"

    # test(modelname, test_loader, model_path, nchannels=18)
