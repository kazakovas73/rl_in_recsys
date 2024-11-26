from datetime import datetime
import os
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import pytz
import torch
from pathlib import Path

from src.policy.onestage import OneStagePolicy, OneStageHyperPolicy_with_DotScore
from src.crititc.qcritic import QCritic
from src.buffer.hyperactor import HyperActorBuffer
from src.simulator.krmb import KRMBUserResponse
from src.reader.krmb import KRMBSeqReader
from src.agent.ddpg import DDPG
from src.agent.hac import HAC
from src.utils import set_random_seed
from src.reward import get_immediate_reward

from src.crititc.sac import SoftQNetwork
from src.policy.sac import SActor

from src.environment.wholesession import KREnvironment_WholeSession_GPU

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="actor_critic", version_base="1.1")
def main(cfg: DictConfig):
    # Starting logging
    logging.basicConfig(level=cfg.logging.level)
    # logger.info(OmegaConf.to_yaml(cfg))

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    seed = cfg.seed
    cuda = cfg.cuda
    path_to_simulator_ckpt = str(Path(cfg.base_dir) / cfg.path_to_simulator_ckpt)
    model_path = output_dir / "model"
    uirm_log_path = output_dir / "log"
    save_path = output_dir / "model"

    set_random_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        torch.cuda.set_device(cuda)
        device = f"cuda"
    else:
        device = "cpu"

    checkpoint = torch.load(path_to_simulator_ckpt + '.checkpoint', map_location=device)
    reader_stats = checkpoint["reader_stats"]

    reader = KRMBSeqReader(
        **cfg.reader
    )

    cfg.simulator.model_path = path_to_simulator_ckpt

    simulator = KRMBUserResponse(
        **cfg.simulator,
        reader_stats=reader_stats
    )

    env = KREnvironment_WholeSession_GPU(
        max_step_per_episode=20, 
        initial_temper=20, 
        device=device,

        uirm_log_path=uirm_log_path,
        slate_size=6,
        episode_batch_size=32,
        item_correlation=0.2,
        single_response=True,

        reader=reader,
        model_path=path_to_simulator_ckpt,
        model=simulator,
        reader_stats=reader_stats,

        from_load=True
    )

    buffer = HyperActorBuffer(
        buffer_size=100_000,
        device=device
    )

    max_action = float(envs.single_action_space.high[0])



    actor = SActor(
        state_user_latent_dim=16,
        state_item_latent_dim=16,
        
    )
    actor.to(device)


    try:
        agent.train()
    except KeyboardInterrupt:
        logger.info("Early stop manually")
        exit_here = input("Exit completely without evaluation? (y/n) (default n):")
        if exit_here.lower().startswith('y'):
            logger.info(os.linesep + '-' * 20 + ' END: ' + datetime.now(pytz.timezone('Europe/Moscow')) + ' ' + '-' * 20)
            exit(1)


if __name__ == "__main__":
    main()
