import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils
from model.agent.BaseRLAgent import BaseRLAgent
from tqdm import tqdm
import wandb


class DDPG(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - critic_lr
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - reward_func
            - n_iter
            - train_every_n_step
            - start_policy_train_at_step
            - initial_epsilon
            - final_epsilon
            - elbow_epsilon
            - explore_rate
            - do_explore_in_train
            - check_episode
            - save_episode
            - save_path
            - actor_lr
            - actor_decay
            - batch_size
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        return parser
    
    
    def __init__(self, *input_args):
        '''
        components:
        - critic
        - critic_optimizer
        - actor_target
        - critic_target
        - components from BaseRLAgent:
            - env
            - actor
            - actor_optimizer
            - buffer
            - exploration_scheduler
            - registered_models
        '''
        args, env, actor, critic, buffer = input_args
        super().__init__(args, env, actor, buffer)

        self.args = args
        
        self.critic_lr = args.critic_lr
        self.critic_decay = args.critic_decay
        self.tau = args.target_mitigate_coef
        
        # models
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
        
        # controller
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)
        self.do_actor_update = True
        self.do_critic_update = True

        # register models that will be saved
        self.registered_models.append((self.critic, self.critic_optimizer, "_critic"))

        
    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        '''
        super().setup_monitors()
        self.training_history.update({'actor_loss': [], 'critic_loss': [], 'Q': [], 'next_Q': []})
        
    def train(self):
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        print("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t

        wandb.finish()
        wandb.init(
            project=self.env.__class__.__name__,
            name=f"{self.__class__.__name__}_{self.actor.__class__.__name__}_{self.critic.__class__.__name__}",
            config={
                **vars(self.args)
            }
        )
        
        # training
        print("Training:")
        step_offset = sum(self.n_iter[:-1])
        do_buffer_update = True
        observation = deepcopy(self.env.current_observation)
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1]//10)):
            do_explore = np.random.random() < self.explore_rate if self.explore_rate < 1 else True
            # online inference
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, 
                                                do_buffer_update, do_explore)
            # online training
            if i % self.train_every_n_step == 0:
                self.step_train()
            # log monitor records
            if i > 0 and i % self.check_episode == 0:
                t_prime = time.time()
                print(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report(smoothness = self.check_episode)
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                print(log_str)
                t = t_prime

                wandb.log({
                    "step": i, 
                    **episode_report,
                    **train_report
                })

            # save model and training info
            if i % self.save_episode == 0:
                self.save()

        wandb.finish()
               
        self.action_after_train()

    
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
        
        # DDPG loss
        
        # Get current Q estimate
        current_critic_output = self.apply_critic(observation, policy_output, self.critic)
        # (B, )
        current_Q = current_critic_output['q']
        
        # Compute the target Q value
        next_policy_output = self.apply_policy(next_observation, self.actor_target, 
                                               0., False, is_train)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        next_Q = target_critic_output['q']
        # (B, )
        target_Q = reward + self.gamma * (done_mask * next_Q).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        
        # Regularization loss
#         critic_reg = current_critic_output['reg']

        if self.do_critic_update and self.critic_lr > 0:
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Compute actor loss
        policy_output = self.apply_policy(observation, self.actor, 
                                          0., self.do_explore_in_train, is_train)
        critic_output = self.apply_critic(observation, policy_output, self.critic)
        actor_loss = -critic_output['q'].mean()
        
        # Regularization loss
#         actor_reg = policy_output['reg']

        if self.do_actor_update and self.actor_lr > 0:
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        loss_dict = {'actor_loss': actor_loss.item(), 
                     'critic_loss': critic_loss.item(), 
                     'Q': torch.mean(current_Q).item(), 
                     'next_Q': torch.mean(next_Q).item()}
    
        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])
        
        return loss_dict
    
    def apply_policy(self, observation, actor, *policy_args):
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
        epsilon = policy_args[0]
        do_explore = policy_args[1]
        is_train = policy_args[2]
        input_dict = {'observation': observation, 
                      'candidates': self.env.get_candidate_info(observation), 
                      'epsilon': epsilon, 
                      'do_explore': do_explore, 
                      'is_train': is_train, 
                      'batch_wise': False}
        out_dict = self.actor(input_dict)
        return out_dict
    
    def apply_critic(self, observation, policy_output, critic):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['hyper_action']}
        return critic(feed_dict)

    def save(self):
        super().save()

    def load(self):
        super().load()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)
