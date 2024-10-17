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
from src.utils import set_random_seed
from src.reward import get_immediate_reward

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
    path_to_simulator_ckpt = "/Users/alexander.kazakov/Documents/rl_in_recsys/output/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model"
    model_path = output_dir / "model"
    uirm_log_path = output_dir / "log"
    save_path = output_dir / "model"

    set_random_seed(seed)

    if cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
        torch.cuda.set_device(cuda)
        device = f"cuda:{cuda}"
    else:
        device = "cpu"

    checkpoint = torch.load(path_to_simulator_ckpt + '.checkpoint', map_location=device)
    reader_stats = checkpoint["reader_stats"]

    reader = instantiate(cfg.reader)

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

    policy = OneStageHyperPolicy_with_DotScore(
        model_path=str(model_path), 
        loss='bce', 
        l2_coef=0.0, 

        state_user_latent_dim=16,
        state_item_latent_dim=16,
        state_transformer_enc_dim=32,
        state_transformer_n_head=4,
        state_transformer_d_forward=64,
        state_transformer_n_layer=3,
        state_dropout_rate=0.1,

        policy_noise_var=0.1,
        policy_noise_clip=1.0,
        policy_do_effect_action_explore=False,
        policy_action_hidden=[256, 64],

        device=device,
        env=env
    )
    policy.to(device)

    critic = QCritic(
        critic_hidden_dims=[256, 64],
        critic_dropout_rate=0.1,
        policy=policy
    )
    critic.to(device)

    buffer = HyperActorBuffer(
        buffer_size=100_000,
        device=device
    )

    agent = DDPG(
        gamma=0.9,
        reward_func=get_immediate_reward,
        n_iter=[20_000],
        train_every_n_step=1,
        start_policy_train_at_step=100,
        initial_epsilon=0.01,
        final_epsilon=0.01,
        elbow_epsilon=0.1,
        explore_rate=1.0,
        do_explore_in_train=False,
        check_episode=10,
        save_episode=200,
        save_path=str(save_path),
        actor_lr=0.0001,
        actor_decay=0.00001,
        batch_size=32,
        
        critic_lr=0.001,
        critic_decay=0.00001,
        target_mitigate_coef=0.01,

        device=device,

        env=env, 
        actor=policy, 
        critic=critic, 
        buffer=buffer
    )


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
