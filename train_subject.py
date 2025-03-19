# import os
import sys
from os.path import dirname, join
from os import rename
from shutil import rmtree

import matplotlib.pyplot as plt
import dataLoader
from train import train
from test import test
import pandas as pd


if __name__ == '__main__':
    args = sys.argv[1:]
    result_df = pd.DataFrame(columns=['model', 'subject', 'filter', 'classes', 'augmentation', 'test_loss',
                                      'test_acc', 'test_diff'])

    subject = f"S{args[0]}"
    session = f"Session{args[1]}"

    LR = 2e-4
    EPOCHS = 300
    batch_size =24
    use_filter = True
    augment = 1e-3

    data_path = join("data", subject, '*')
    model_name = 'EEGNet'

    for classes in [['MM', 'MI'], ['MI', 'Rest']]:
        output_path = join("output", subject, session)

        if classes[0] == 'MM':
            posweight = 2
        else:
            posweight = 1.5

        train_loader, val_loader, test_loader, full_train_loader = dataLoader.load(data_path, batch_size,
                                                                                   augment=augment, pos=classes[0],
                                                                                   neg=classes[1])

        train_result, valid_result, train_result_acc, val_result_acc, version, epochs = train(model_name, train_loader,
                                                                                              val_loader, LR, EPOCHS,
                                                                                              batch_size, output_path,
                                                                                              pos_weight=posweight)

        best_epoch = valid_result.index(min(valid_result)) + 1
        best_epoch_train_result = train_result[best_epoch-1]
        model_path = join(output_path, "models", version, f"{model_name}_{version}_epoch_{best_epoch}.pt")

        full_train_result, _, full_train_result_acc, _, full_train_version, full_train_epochs = train(model_name, full_train_loader,
                                                                                                      None, LR, EPOCHS,
                                                                                                      batch_size, output_path,
                                                                                                      validate=False, model_path=model_path,
                                                                                                      val_loss=best_epoch_train_result,
                                                                                                      pos_weight=posweight)

        rmtree(dirname(model_path))
        log_path = join(output_path, "logs", full_train_version + "_retrain")

        model_path = join(output_path, "models", full_train_version + "_retrain", f"{model_name}_{full_train_version}_epoch_{full_train_epochs}.pt")

        test_acc, avg_test_loss, avg_diff = test(model_name, test_loader, model_path, pos_weight=posweight)

        result = pd.DataFrame({'model': model_name, 'subject': subject, 'filter': use_filter, 'classes': str(classes),
                               'augmentation': augment, 'test_loss': avg_test_loss, 'test_acc': test_acc,
                               'test_diff': avg_diff}, index=[0])

        result.to_csv(join(log_path, "test_result.csv"))

        if classes[0] == 'MM':
            rename(model_path, join(output_path, f"mimm_{subject}_{session}.pt"))
        if classes[0] == 'MI':
            rename(model_path, join(output_path, f"restmi_{subject}_{session}.pt"))

        result_df = pd.concat([result_df, result], ignore_index=True)

        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, epochs+1), train_result)
        ax1.plot(range(best_epoch+1, full_train_epochs+best_epoch+1), full_train_result, 'b.-')
        ax1.plot(range(1, epochs+1), valid_result)
        ax1.plot(best_epoch+full_train_epochs, avg_test_loss, 'r*', markersize = 12)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Train vs Validation Loss')
        ax1.legend(['Train', 'Full train', 'Validation', 'Test'])
        fig1.savefig(join(log_path, "loss.png"))

        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, epochs+1), train_result_acc)
        ax2.plot(range(best_epoch+1, full_train_epochs+best_epoch+1), full_train_result_acc, 'b.-')
        ax2.plot(range(1, epochs+1), val_result_acc)
        ax2.plot(best_epoch+full_train_epochs, test_acc, 'r*', markersize = 12)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Train vs Validation Accuracy')
        ax2.legend(['Train', 'Full train', 'Validation', 'Test'])
        fig2.savefig(join(log_path, "accuracy.png"))
        result_df.to_csv(f'train_results_subject{subject}.csv')