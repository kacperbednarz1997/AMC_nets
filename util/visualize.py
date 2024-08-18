from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import scipy.io

def Visualize_LiftingScheme(model, signal, cfg, index):
    def hook_feature_map(module, *input_feature_map):
        if len(feature_map) < 3:
            feature_map.append(input_feature_map[0][0])
        feature_map.append(input_feature_map[1][0])  # approx
        feature_map.append(input_feature_map[1][1])  # details

    feature_map = []

    for i in range(0, cfg.num_level):
        suffix = f'levels.level_{i}.wavelet.register_forward_hook'
        split = suffix.split('.')
        reduce(getattr, split, model)(hook=hook_feature_map)

    model.to('cpu')
    model(signal)

    fig1 = plt.figure(figsize=(6, 10))
    gs = GridSpec(cfg.num_level + 1, 2, figure=fig1)

    for i in range(0, len(feature_map)):
        if i == 0:
            ax = fig1.add_subplot(gs[i, :2])
            ax.plot(feature_map[i].detach().numpy()[0, 0, :])
            ax.set_title(fr"$V_0$")
        else:
            ax = fig1.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
            ax.plot(feature_map[i].detach().numpy()[0, 0, :])
            if (i + 1) % 2 == 0:
                ax.set_title(fr'$L_{(i + 1) // 2}$')
            else:
                ax.set_title(fr'$H_{(i + 1) // 2}$')
        ax.tick_params(axis='both', which='major', labelsize=14)  # Set font size for axis tick labels

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig1.savefig(cfg.result_dir + '/' + f'visualize_one_channel_#{index}.svg', format='svg', dpi=150)

    fig2 = plt.figure(figsize=(6, 10))
    gs = GridSpec(cfg.num_level + 1, 2, figure=fig2)

    for i in range(0, len(feature_map)):
        if i == 0:
            ax = fig2.add_subplot(gs[i, :2])
            ax.imshow(feature_map[i].detach().numpy()[0], aspect='auto', cmap='RdBu')
            ax.set_title(fr"$V_0$")
        else:
            ax = fig2.add_subplot(gs[(i + 1) // 2, (i + 1) % 2])
            ax.imshow(feature_map[i].detach().numpy()[0], aspect='auto', cmap='RdBu')
            if (i + 1) % 2 == 0:
                ax.set_title(fr'$L_{(i + 1) // 2}$')
            else:
                ax.set_title(fr'$H_{(i + 1) // 2}$')
        ax.set_yticks([])
        ax.tick_params(axis='both', which='major', labelsize=14)  # Set font size for axis tick labels

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    fig2.savefig(cfg.result_dir + '/' + f'visualize_feature_map_#{index}.svg', format='svg', dpi=150)

    plt.close()

def Draw_Confmat(Confmat_Set, snrs, cfg):
    for i, snr in enumerate(snrs):
        if snr == 'ota_1m':
            snr = '37'
        elif snr == 'ota_6m':
            snr = '22'
        else:
            snr = snr
        fig = plt.figure(figsize=(15, 15))  # Increased figsize to make the plot larger
        classes = {key.decode('utf-8'): value for key, value in cfg.classes.items()}

        df_cm = pd.DataFrame(Confmat_Set[i], index=classes, columns=classes)
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True, annot_kws={"size": 13})
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
        plt.ylabel('True label', fontsize=16)
        plt.xlabel('Predicted label', fontsize=16)
        plt.tight_layout()

        conf_mat_dir = os.path.join(cfg.result_dir, 'conf_mat')
        os.makedirs(conf_mat_dir, exist_ok=True)
        fig.savefig(conf_mat_dir + '/' + f'ConfMat_{snr}dB.svg', format='svg', dpi=150)
        plt.close()

def Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg, file_path):
    for i, snr in enumerate(snrs):
        if snr == 'ota_1m':
            snrs[i] = '37'
        elif snr == 'ota_6m':
            snrs[i] = '22'
        else:
            snrs[i] = snrs[i]

    # Inicjalizacja list, które będą przechowywać dane
    snr_all = []
    accuracy_all = []
    # Sprawdzenie, czy plik .mat istnieje
    if os.path.exists(file_path):
        # Wczytanie danych z pliku .mat
        data = scipy.io.loadmat(file_path)
        snr_all = data.get('snr_all', []).flatten().tolist()
        accuracy_all = data.get('accuracy_all', []).flatten().tolist()
    # Dodanie nowych danych do list
    snr_all.extend(snrs)
    accuracy_all.extend(Accuracy_list)
    # Konwersja do numpy array przed zapisem
    snr_all = np.array(snr_all)
    accuracy_all = np.array(accuracy_all)
    # Zapisanie danych do pliku .mat
    scipy.io.savemat(file_path, {'snr_all': snr_all, 'accuracy_all': accuracy_all})

    plt.figure(figsize=(12, 8))  # Increased figure size
    plt.plot(snrs, Accuracy_list)
    plt.xlabel("Signal to Noise Ratio [dB]", fontsize=16)
    plt.ylabel("Overall Accuracy", fontsize=16)
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    acc_dir = os.path.join(cfg.result_dir, 'acc')
    os.makedirs(acc_dir, exist_ok=True)
    #plt.legend(loc='best')  # Position the legend in the best location
    plt.savefig(acc_dir + '/' + 'acc.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.close()

    Accuracy_Mods = np.zeros((len(snrs), Confmat_Set.shape[-1]))

    for i, snr in enumerate(snrs):
        Accuracy_Mods[i, :] = np.diagonal(Confmat_Set[i]) / Confmat_Set[i].sum(1)

    plt.figure(figsize=(12, 8))  # Increased figure size
    for j in range(0, Confmat_Set.shape[-1]):
        plt.plot(snrs, Accuracy_Mods[:, j], label=f'Class {j}')

    plt.xlabel("Signal to Noise Ratio [dB]", fontsize=16)
    plt.ylabel("Overall Accuracy", fontsize=16)
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    classes = {key.decode('utf-8'): value for key, value in cfg.classes.items()}
    plt.legend(classes.keys(), loc='best')  # Position the legend in the best location
    plt.savefig(acc_dir + '/' + 'acc_mods.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.close()

def save_training_process(train_process, cfg):
    fig1 = plt.figure(figsize=(8, 6))  # Increased figure size for better visibility
    plt.plot(train_process.epoch, train_process.lr_list)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Learning Rate", fontsize=16)  # Increased font size for better readability
    #plt.title("Learning Rate", fontsize=16)  # Increased font size for better readability
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Adjust layout to fit everything properly
    fig1.savefig(cfg.result_dir + '/' + 'lr.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.close()

    fig2 = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss, "ro-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss, "bs-", label="Val loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)  # Increased font size for better readability
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc, "ro-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc, "bs-", label="Val acc")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)  # Increased font size for better readability
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=14)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Set font size for y-axis tick labels
    plt.tight_layout()  # Adjust layout to fit everything properly
    fig2.savefig(cfg.result_dir + '/' + 'loss_acc.svg', format='svg', dpi=150, bbox_inches='tight')  # Ensure the entire plot is saved
    plt.show()
    plt.close()
