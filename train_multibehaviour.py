import os
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
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

logger = logging.getLogger(__name__)

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

@hydra.main(config_path="conf", config_name="multibehaviour", version_base="1.1")
def main(cfg: DictConfig):
    # Starting logging
    logging.basicConfig(level=cfg.logging.level)
    # logger.info(OmegaConf.to_yaml(cfg))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    torch.multiprocessing.set_sharing_strategy('file_system')

    cuda = cfg.cuda
    seed = cfg.seed
    lr = cfg.lr
    batch_size = cfg.batch_size
    val_batch_size = cfg.val_batch_size
    epochs = cfg.epochs
    save_with_val = cfg.save_with_val

    set_random_seed(seed)

    if cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        torch.cuda.set_device(cuda)
        device = f"cuda:{cuda}"
    else:
        device = "cpu"

    reader = KRMBSeqReader(**cfg.reader)

    model = KRMBUserResponse(
        **cfg.simulator,
        reader_stats=reader.get_statistics(),
        logger=logger
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.optimizer = optimizer


    try:
        
        best_auc = {f: 0 for f in model.feedback_types}
        logger.info(f"validation before training:")

        val_report = do_eval(model, reader, device, batch_size, val_batch_size)

        logger.info(f"Val result:")
        logger.info(val_report)
        
        epo = 0
        stop_count = 0
        while epo < epochs:
            epo += 1
            logger.info(f"epoch {epo} training")
            
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
                    logger.info(f"Iteration {i}, loss: {np.mean(step_loss[-100:])}")
                    logger.info({fb: np.mean(v[-100:]) for fb,v in step_behavior_loss.items()})
            pbar.close()
            logger.info("Epoch {}; time {:.4f}".format(epo, time() - t1))

            # validation
            t2 = time()
            logger.info(f"epoch {epo} validating")
            val_report = do_eval(model, reader, device, batch_size, val_batch_size)
            logger.info(f"Val result:")
            logger.info(val_report)
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
        logger.info("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            print(os.linesep + '-' * 20 + ' END: ' + datetime.now(pytz.timezone('Europe/Moscow')) + ' ' + '-' * 20)
            exit(1)


if __name__ == "__main__":
    main()
