# import pybullet_envs
# import gym
from copy import deepcopy
import logging
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
# from gym import wrappers
import torch as T

from buffer import ReplayBuffer, BaseBuffer
from encoder import Encoder

from src.environment.wholesession import KREnvironment_WholeSession_GPU
from src.simulator.krmb import KRMBUserResponse
from src.reader.krmb import KRMBSeqReader
from src.reward import get_immediate_reward

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    device = 'cuda:0' if T.cuda.is_available() else 'cpu'

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
        data_separator = ','
    )

    model_path = "env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model"
    simulator = KRMBUserResponse(
        model_path = model_path,
        loss = "bce",
        l2_coef = 0.0,

        user_latent_dim = 32,
        item_latent_dim = 32,
        enc_dim = 64,
        attn_n_head = 4,
        transformer_d_forward = 64,
        transformer_n_layer = 2,
        state_hidden_dims = [128],
        scorer_hidden_dims = [128, 32],
        dropout_rate = 0.1,

        reader_stats=reader.get_statistics(),
        logger=logger,
        device=device
    )

    slate_size = 6

    env = KREnvironment_WholeSession_GPU(
        max_step_per_episode=20, 
        initial_temper=20, 
        device=device,

        uirm_log_path="log/",
        slate_size=slate_size,
        episode_batch_size=4,
        item_correlation=0.2,
        single_response=True,

        reader=reader,
        model_path=model_path,
        model=simulator,
        reader_stats=reader.get_statistics(),

        from_load=False
    )

    state_user_latent_dim = 16
    state_item_latent_dim = 16
    enc_dim = 32
    state_dim = 3 * enc_dim

    action_dim = 32

    encoder = Encoder(
        model_path=model_path,
        loss='bce',
        l2_coef=0.0, 
        state_user_latent_dim=state_user_latent_dim,
        state_item_latent_dim=state_item_latent_dim,
        state_transformer_enc_dim=enc_dim,
        state_transformer_n_head=4,
        state_transformer_d_forward=64,
        state_transformer_n_layer=3,
        state_dropout_rate=0.1,

        device=device,
        reader_stats=reader.get_statistics(),
        logger=logger
    )

    buffer = BaseBuffer(
        buffer_size=100,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim
    )

    agent = Agent(
        encoder=encoder,
        buffer=buffer,
        input_dims=[state_dim], 
        env=env,
        action_dim=action_dim,
        item_dim=enc_dim,
        slate_size=slate_size
    )
    n_games = 250
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    # filename = 'recsys.png'
    # figure_file = 'plots/' + filename

    # best_score = 0
    # score_history = []
    # load_checkpoint = False

    # if load_checkpoint:
    #     agent.load_models()
    #     env.render(mode='human')

    # before training
    env.reset()
    buffer.reset(env)
    

    reward_func = get_immediate_reward
    def get_reward(user_feedback):
        user_feedback['immediate_response_weight'] = env.response_weights
        R = reward_func(user_feedback).detach()
        return R

    observation = deepcopy(env.current_observation)
    for i in range(50):
        
        done = False
        current_sum_reward = T.zeros(env.episode_batch_size).to(T.float).to(device)

        with T.no_grad():
            input_dict = {
                'observation': observation, 
                'candidates': env.get_candidate_info(observation), 
                # 'epsilon': epsilon, 
                # 'do_explore': do_explore, 
                # 'is_train': is_train, 
                'batch_wise': False
            }
            
            action = agent.choose_action(input_dict)
            action_dict = {'action': action['indices']}
            new_observation, user_feedback, update_info = env.step(action_dict)

            R = get_reward(user_feedback)
            user_feedback['reward'] = R
            current_sum_reward = current_sum_reward + R
            done_mask = user_feedback['done']

        if T.sum(done_mask) > 0:
            print(f"avg_total_reward: {current_sum_reward[done_mask].mean().item()}")
            # print(f"max_total_reward: {current_sum_reward[done_mask].max().item()}")
            # print(f"min_total_reward: {current_sum_reward[done_mask].min().item()}")
            current_sum_reward[done_mask] = 0

        # print(f"avg_reward: {R.mean().item()}")
        # print(f"reward_variance: {T.var(R).item()}")

        # for i,resp in enumerate(env.response_types):
        #     print(f"{resp}_rate: {user_feedback['immediate_response'][:,:,i].mean().item()}")

        buffer.update(observation, action, user_feedback, update_info['updated_observation'])

        observation = new_observation

        agent.learn(env)

    # env.stop()



    #         score += reward
    #         agent.remember(observation, action, reward, observation_, done)
    #         if not load_checkpoint:
    #             agent.learn()
    #         observation = observation_
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-100:])

    #     if avg_score > best_score:
    #         best_score = avg_score
    #         if not load_checkpoint:
    #             agent.save_models()

    #     print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)