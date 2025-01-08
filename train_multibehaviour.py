import os
from pathlib import Path
import torch
from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from src.simulator.krmb import KRMBUserResponse
from src.reader.krmb import KRMBSeqReader

from datetime import datetime
import pytz

from src.utils import set_random_seed, wrap_batch, show_batch

def do_eval(model, reader, device, batch_size, val_batch_size):
    reader.set_phase("val")
    eval_loader = DataLoader(reader, batch_size = val_batch_size,
                             shuffle = False, pin_memory = False, 
                             num_workers = reader.n_worker)
    val_report = {'loss': [], 'auc': {}}
    Y_dict = {f: [] for f in model.feedback_types}
    P_dict = {f: [] for f in model.feedback_types}
    pbar = tqdm(total = len(reader))
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            wrapped_batch = wrap_batch(batch_data, device = device)
            out_dict = model.do_forward_and_loss(wrapped_batch)
            loss = out_dict['loss']
            val_report['loss'].append(loss.item())
            for j,f in enumerate(model.feedback_types):
                Y_dict[f].append(wrapped_batch[f].view(-1).detach().cpu().numpy())
                P_dict[f].append(out_dict['preds'][:,:,j].view(-1).detach().cpu().numpy())
            pbar.update(batch_size)
    val_report['loss'] = (np.mean(val_report['loss']), np.min(val_report['loss']), np.max(val_report['loss']))
    for f in model.feedback_types:
        val_report['auc'][f] = roc_auc_score(np.concatenate(Y_dict[f]), 
                                             np.concatenate(P_dict[f]))
    pbar.close()
    return val_report

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    output_dir = Path("env/")

    cuda = 0
    seed = 42
    lr = 0.0001
    reg_coef = 0.001
    batch_size = 128
    val_batch_size = 128
    epochs = 2
    save_with_val = True

    set_random_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        torch.cuda.set_device(cuda)
        device = f"cuda:{cuda}"
    else:
        device = "cpu"

    reader = KRMBSeqReader(
        user_meta_file = "dataset/user_features_Pure_fillna.csv",
        item_meta_file = "dataset/video_features_basic_Pure_fillna.csv",
        max_hist_seq_len = 100,
        val_holdout_per_user = 5,
        test_holdout_per_user = 5,
        meta_file_sep = ',',
        train_file = "dataset/log_session_4_08_to_5_08_Pure.csv",
        val_file = '',
        test_file = '',
        n_worker = 4,
        data_separator = ',',
    )

    model = KRMBUserResponse(
        model_path = f"env/user_KRMBUserResponse_lr0.0001_reg0.001_nlayer2.model",
        loss = 'bce',
        l2_coef = reg_coef,

        user_latent_dim = 32,
        item_latent_dim = 32,
        enc_dim = 64,
        attn_n_head = 4,
        transformer_d_forward = 64,
        transformer_n_layer = 2,
        state_hidden_dims = [128],
        scorer_hidden_dims = [128, 32],
        dropout_rate = 0.1,
        device = device, 
        reader_stats=reader.get_statistics()
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.optimizer = optimizer


    try:
        
        best_auc = {f: 0 for f in model.feedback_types}
        print(f"validation before training:")

        val_report = do_eval(model, reader, device, batch_size, val_batch_size)
        print(f"Val result:")
        print(val_report)
        
        epo = 0
        stop_count = 0
        while epo < epochs:
            epo += 1
            print(f"epoch {epo} training")
            
            # train an epoch
            model.train()
            reader.set_phase("train")
            train_loader = DataLoader(reader, batch_size = batch_size,
                                      shuffle = True, pin_memory = True,
                                      num_workers = reader.n_worker)
            t1 = time()
            pbar = tqdm(total = len(reader))
            step_loss = []
            step_behavior_loss = {fb: [] for fb in model.feedback_types}
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                wrapped_batch = wrap_batch(batch_data, device = device)
                if epo == 1 and i == 0:
                    show_batch(wrapped_batch)
                out_dict = model.do_forward_and_loss(wrapped_batch)
                loss = out_dict['loss']
                loss.backward()
                step_loss.append(loss.item())
                for fb, v in out_dict['behavior_loss'].items():
                    step_behavior_loss[fb].append(v)
                optimizer.step()
                pbar.update(batch_size)
                if i % 100 == 0:
                    print(f"Iteration {i}, loss: {np.mean(step_loss[-100:])}")
                    print({fb: np.mean(v[-100:]) for fb,v in step_behavior_loss.items()})
            pbar.close()
            print("Epoch {}; time {:.4f}".format(epo, time() - t1))

            # validation
            t2 = time()
            print(f"epoch {epo} validating")
            val_report = do_eval(model, reader, device, batch_size, val_batch_size)
            print(f"Val result:")
            print(val_report)
            improve = 0
            for f,v in val_report['auc'].items():
                if v > best_auc[f]:
                    improve += 1
                    best_auc[f] = v

            # save model when no less than 50% of the feedback types are improved
            if save_with_val:
                if improve >= 0.5 * len(model.feedback_types):
                    model.save_checkpoint(output_dir)
                    stop_count = 0
                else:
                    stop_count += 1
                if stop_count >= 3:
                    break
            else:
                model.save_checkpoint(output_dir)
            
    except KeyboardInterrupt:
        print("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + datetime.now(pytz.timezone('Europe/Moscow')) + ' ' + '-' * 20)
            exit(1)


if __name__ == "__main__":
    main()