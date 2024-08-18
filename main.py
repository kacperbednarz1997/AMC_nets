import argparse
import os

import numpy as np
import torch

from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.evaluation import Run_Eval
from util.training import Trainer
from util.utils import fix_seed, log_exp_settings, create_AWN_model, create_CNN_model, create_CLDNN_model, create_CGDNN_model
from util.utils import create_Resnet_model, create_RNN_model, create_LSTM_model, create_AMC_Net
from util.logger import create_logger
from util.visualize import Visualize_LiftingScheme, save_training_process


if __name__ == "__main__":
    #models = ["CNN_Pijackova", "RNN", "LSTM", "CLDNN_Pijackova", "CGDNN_Pijackova", "Resnet_Alexivaner", "AWN_model"]
    models = ["AWN_model"]
    #datasets = ["2016.10a", "2016.10b", "2018.01a", "migou_dataset_19.08"]
    datasets = ["2016.10a"]
    mode = 'eval' # train ,eval or visualize
    file_path = 'snr_acc_list.mat'
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Plik {file_path} został usunięty.")
    else:
        print(f"Plik {file_path} nie istnieje.")
    for tmp_model_name in models:
        for tmp_dataset_name in datasets:
            #time.sleep(120)  # sleep 120s * x
            parser = argparse.ArgumentParser()
            parser.add_argument('--mode', type=str, default=mode)  # train ,eval or visualize
            parser.add_argument('--dataset', type=str, default=tmp_dataset_name)  # 2016.10a, 2016.10b, 2018.01a, migou_dataset_19.08
            parser.add_argument('--model_name', type=str, default=tmp_model_name) # "CNN_Pijackova", "RNN", "LSTM", "CLDNN_Pijackova", "CGDNN_Pijackova", "Resnet_Alexivaner", "AWN_model"
            #CNN_OShea, Resnet_Alexivaner, VT_CNN2, RNN, LSTM, AMCNet, AWN_modified
            parser.add_argument('--seed', type=int, default=2022)
            parser.add_argument('--device', type=str,
                                default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
            parser.add_argument('--num_workers', type=int, default=0)
            parser.add_argument('--Draw_Confmat', type=bool, default=True)
            parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)

            args = parser.parse_args()

            fix_seed(args.seed)

            cfg = Config(args.dataset,args.model_name, train=(args.mode == 'train'))
            cfg = merge_args2cfg(cfg, vars(args))
            logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
            log_exp_settings(logger, cfg)

            if args.model_name == "AWN_model":
                model = create_AWN_model(cfg)
            elif args.model_name == "CNN_Pijackova":
                model = create_CNN_model(cfg)
            elif args.model_name == "CLDNN_Pijackova":
                model = create_CLDNN_model(cfg)
            elif args.model_name == "CGDNN_Pijackova":
                model = create_CGDNN_model(cfg)
            elif args.model_name == "Resnet_Alexivaner":
                model = create_Resnet_model(cfg)
            elif args.model_name == "RNN":
                model = create_RNN_model(cfg)
            elif args.model_name == "LSTM":
                model = create_LSTM_model(cfg)
            elif args.model_name == "AMCNet":
                model = create_AMC_Net(cfg)
            else:
                print("Nie ma takiego modelu!")
            #print(model)
            #if args.dataset == "2018.01a":
            #    in_size = (2, 1024)
            #else:
            #    in_size = (2, 128)
            #summary(model, in_size)

            logger.info(">>> total params: {:.2f}M".format(
                sum(p.numel() for p in list(model.parameters())) / 1000000.0))

            Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger)
            train_set, test_set, val_set, test_idx = Dataset_Split(
                Signals,
                Labels,
                snrs,
                mods,
                logger)
            Signals_test, Labels_test = test_set

            if args.mode == 'train':
                train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
                trainer = Trainer(model,
                                  train_loader,
                                  val_loader,
                                  cfg,
                                  logger,
                                  cfg.optim_flag,
                                  args.model_name)
                trainer.loop()

                save_training_process(trainer.epochs_stats, cfg)

                save_model_name = cfg.dataset + '_' + args.model_name + '.pkl'
                model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name)))
                Run_Eval(model,
                         Signals_test,
                         Labels_test,
                         SNRs,
                         test_idx,
                         cfg,
                         logger)

            elif args.mode == 'eval':
                model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + args.model_name + '.pkl')))
                Run_Eval(model,
                         Signals_test,
                         Labels_test,
                         SNRs,
                         test_idx,
                         cfg,
                         logger,
                         file_path)

            elif args.mode == 'visualize':
                model.load_state_dict(torch.load(os.path.join(args.ckpt_path, cfg.dataset + '_' + args.model_name + '.pkl')))
                for i in range(0, 8):
                    index = np.random.randint(0, Signals_test.shape[0])
                    test_sample = Signals_test[index]
                    test_sample = test_sample[np.newaxis, ...]
                    Visualize_LiftingScheme(model, test_sample, cfg, index)