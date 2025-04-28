import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from src.agent.base import BaseRLAgent
from src.utils import LinearScheduler

from src.crititc.qcritic import QCritic, ValueNetwork
from src.policy.sac import Actor



class SAC(BaseRLAgent):
    def __init__(
        self,

        gamma,
        reward_func,
        n_iter,
        train_every_n_step,
        start_policy_train_at_step,
        initial_epsilon,
        final_epsilon,
        elbow_epsilon,
        explore_rate,
        do_explore_in_train,
        check_episode,
        save_episode,
        save_path,
        actor_lr,
        actor_decay,
        batch_size,
        
        critic_lr,
        critic_decay,
        target_mitigate_coef,

        device,


        env,
        actor,
        buffer,
        logger  
    ):
        super().__init__(
            reward_func,
            n_iter,
            train_every_n_step,
            start_policy_train_at_step,
            initial_epsilon,
            final_epsilon,
            elbow_epsilon,
            explore_rate,
            do_explore_in_train,
            check_episode,
            save_episode,
            save_path,
            actor_lr,
            actor_decay,
            batch_size,
            device,
            env, actor, buffer,
            logger)

        self.gamma = gamma

        self.critic_lr = critic_lr
        self.critic_decay = critic_decay
        self.tau = target_mitigate_coef

        # models
        self.actor = actor
        self.critic_1 = QCritic(
            critic_hidden_dims=[256, 64],
            critic_dropout_rate=0.1,
            policy=actor,
            logger=logger
        )
        self.critic_2 = QCritic(
            critic_hidden_dims=[256, 64],
            critic_dropout_rate=0.1,
            policy=actor,
            logger=logger
        )
        self.value = ValueNetwork(
            value_hidden_dims=[256, 64],
            value_dropout_rate=0.1,
            policy=actor,
            logger=logger
        )
        self.target_value = ValueNetwork(
            value_hidden_dims=[256, 64],
            value_dropout_rate=0.1,
            policy=actor,
            logger=logger
        )

        self.critic_1.to(device)
        self.critic_2.to(device)
        self.value.to(device)
        self.target_value.to(device)

        # controller
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr, weight_decay=critic_decay)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr, weight_decay=critic_decay)

        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr, weight_decay=critic_decay)
        self.target_value_optimizer = torch.optim.Adam(self.target_value.parameters(), lr=critic_lr, weight_decay=critic_decay)

        self.do_actor_update = True
        self.do_critic_update = True

        # register models that will be saved
        self.registered_models.append((self.critic_1, self.critic_1_optimizer, "_critic_1"))
        self.registered_models.append((self.critic_2, self.critic_2_optimizer, "_critic_2"))

        self.registered_models.append((self.value, self.value_optimizer, "_value"))
        self.registered_models.append((self.target_value, self.target_value_optimizer, "_target_value"))

        self.update_network_parameters(tau=1)

    def step_train(self):
        '''
        @process:
        - get sample
        - calculate Q'(s_{t+1}, a_{t+1}) and Q(s_t, a_t)
        - critic loss: TD error loss
        - critic optimization
        - actor loss: Q(s_t, \pi(s_t)) maximization
        - actor optimization
        '''
        
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        epsilon = 0
        is_train = True
        # (B, )
        reward = user_feedback['reward'].view(-1)


        value = self.value(policy_output).view(-1)

        input_dict = {
            'observation': next_observation, 
            'candidates': self.env.get_candidate_info(next_observation), 
            'epsilon': epsilon, 
            # 'do_explore': do_explore, 
            'is_train': is_train, 
            'batch_wise': False
        }

        next_policy_output = self.actor(input_dict)

        value_ = self.target_value(next_policy_output).view(-1)

        value_[done_mask] = 0.0

        actions, log_probs = self.actor.sample_normal(policy_output['state'], reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward({'state': policy_output['state'], 'action': actions})['q']
        q2_new_policy = self.critic_2.forward({'state': policy_output['state'], 'action': actions})['q']
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        actions, log_probs = self.actor.sample_normal(policy_output['state'], reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward({'state': policy_output['state'], 'action': actions})['q']
        q2_new_policy = self.critic_2.forward({'state': policy_output['state'], 'action': actions})['q']
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        q_hat = 2 * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward({'state': policy_output['state'], 'action': policy_output['action']})['q'].view(-1)
        q2_old_policy = self.critic_2.forward({'state': policy_output['state'], 'action': policy_output['action']})['q'].view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
            if tau is None:
                tau = self.tau

            target_value_params = self.target_value.named_parameters()
            value_params = self.value.named_parameters()

            target_value_state_dict = dict(target_value_params)
            value_state_dict = dict(value_params)

            for name in value_state_dict:
                value_state_dict[name] = tau*value_state_dict[name].clone() + \
                        (1-tau)*target_value_state_dict[name].clone()

            self.target_value.load_state_dict(value_state_dict)


    def action_before_train(self):
        '''
        Action before training:
        - env.reset()
        - buffer.reset()
        - set up training monitors
            - training_history
            - eval_history
        - run several episodes of random actions to build-up the initial buffer
        '''
        
        observation = self.env.reset()
        self.buffer.reset(self.env, self.actor)
        
        # training monitors
        self.setup_monitors()
        
        episode_iter = 0 # zero training iteration
        pre_epsilon = 1.0 # uniform random explore before training
        do_buffer_update = True
        prepare_step = 0
        
        for _ in tqdm(range(self.start_policy_train_at_step)):
            do_explore = np.random.random() < self.explore_rate
            observation = self.run_episode_step(episode_iter, pre_epsilon, observation, 
                                                do_buffer_update, do_explore)
            prepare_step += 1
        print(f"Total {prepare_step} prepare steps")

    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'actor_loss': [], 'critic_loss': [], 'Q': [], 'next_Q': []})


    def run_episode_step(self, i, epsilon, observation, do_buffer_update, do_explore):
        '''
        Run one step of user-env interaction
        @input:
        - episode_args: (episode_iter, epsilon, observation, do_buffer_update, do_explore)
        @process:
        - apply_policy: observation, candidate items --> policy_output
        - env.step(): policy_output['action'] --> user_feedback, updated_observation
        - reward_func(): user_feedback --> reward
        - buffer.update(observation, policy_output, user_feedback, updated_observation)
        @output:
        - next_observation
        '''
        self.epsilon = epsilon
        is_train = False
        with torch.no_grad():
            # generate action from policy
            policy_output = self.apply_policy(observation, self.actor, epsilon, do_explore, is_train)
            
            # apply action on environment
            # Note: action must be indices on env.candidate_iids
            action_dict = {'action': policy_output['indices']}
            new_observation, user_feedback, update_info = self.env.step(action_dict)
            
            # calculate reward
            R = self.get_reward(user_feedback)
            user_feedback['reward'] = R
            self.current_sum_reward = self.current_sum_reward + R
            done_mask = user_feedback['done']
            if torch.sum(done_mask) > 0:
                self.eval_history['avg_total_reward'].append(self.current_sum_reward[done_mask].mean().item())
                self.eval_history['max_total_reward'].append(self.current_sum_reward[done_mask].max().item())
                self.eval_history['min_total_reward'].append(self.current_sum_reward[done_mask].min().item())
                self.current_sum_reward[done_mask] = 0
            
            # monitor update
            self.eval_history['avg_reward'].append(R.mean().item())
            self.eval_history['reward_variance'].append(torch.var(R).item())
            
            for i,resp in enumerate(self.env.response_types):
                self.eval_history[f'{resp}_rate'].append(user_feedback['immediate_response'][:,:,i].mean().item())  
            # update replay buffer
            if do_buffer_update:
                self.buffer.update(observation, policy_output, user_feedback, update_info['updated_observation'])
        return new_observation


    def apply_policy(self, observation, actor, epsilon, do_explore, is_train):
        '''
        @input:
        - observation:{'user_profile':{
                           'user_id': (B,)
                           'uf_{feature_name}': (B,feature_dim), the user features}
                       'user_history':{
                           'history': (B,max_H)
                           'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
        - actor: the actor model
        - epsilon: scalar
        - do_explore: boolean
        - is_train: boolean
        @output:
        - policy_output
        '''
        input_dict = {'observation': observation, 
                      'candidates': self.env.get_candidate_info(observation), 
                      'epsilon': epsilon, 
                      'do_explore': do_explore, 
                      'is_train': is_train, 
                      'batch_wise': False}
        out_dict = actor(input_dict)
        return out_dict
    
    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['hyper_action']}
        return critic(feed_dict)

    def get_reward(self, user_feedback):
        user_feedback['immediate_response_weight'] = self.env.response_weights
        R = self.reward_func(user_feedback).detach()
        return R
