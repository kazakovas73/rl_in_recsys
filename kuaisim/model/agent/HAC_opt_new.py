import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from itertools import chain
from copy import deepcopy
import utils
from model.agent.DDPG import DDPG
from model.components import DNN
from tqdm import tqdm
import wandb


class HAC_opt_new(DDPG):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - behavior_lr
        - behavior_decay
        - hyper_actor_coef
        - hyper_noise_std
        - hyper_noise_clip
        - from DDPG
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
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--hyper_actor_coef', type=float, default=0.1, 
                            help='hyper actor loss coefficient')
        parser.add_argument('--eta', type=float, default=0.1, 
                            help='eta')
        return parser
    
    
    def __init__(self, args, env, actor, critic, potential, buffer):
        assert env.single_response
        super().__init__(*[args, env, actor, critic, buffer])
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.hyper_actor_coef = args.hyper_actor_coef
        
        self.inverse_module = DNN(actor.enc_dim, [256], actor.action_dim, 
                                  dropout_rate = actor.dropout_rate, do_batch_norm = True)
        self.inverse_module = self.inverse_module.to(args.device)
        self.inverse_module_optimizer = torch.optim.Adam(chain(self.inverse_module.parameters(), self.actor.parameters()), 
                                                         lr=args.hyper_actor_coef, weight_decay=args.actor_decay)
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                         lr=args.behavior_lr, weight_decay=args.behavior_decay)

        self.args = args
        
        self.w = args.w
        self.eta = args.eta
        
        self.potential = potential
        self.potential_optimizer = torch.optim.Adam(
            self.potential.parameters(),
            lr=args.critic_lr / 100, 
            # weight_decay=args.behavior_decay
        )

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

    def setup_monitors(self):
        '''
        This is used in super().action_before_train() in super().train()
        Then the agent will call several rounds of run_episode_step() for collecting initial random data samples
        
        DDPG monitors:
        - training_history: actor_loss, critic_loss, Q, nextQ
        
        HAC extra monitors:
        - training_history: hyper_actor_loss, behavior_loss
        '''
        super().setup_monitors()
        self.training_history.update({"hyper_actor_loss": [], 
                                      "behavior_loss": []})
                                    

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

        critic_loss, actor_loss, hyper_actor_loss, behavior_loss = self.get_hac_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['hyper_actor_loss'].append(hyper_actor_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1], 
                              self.training_history['hyper_actor_loss'][-1], 
                              self.training_history['behavior_loss'][-1])}
    
    def get_hac_loss(self, observation, policy_output, user_feedback, 
                     done_mask, next_observation, 
                     do_actor_update = True, do_critic_update = True):
        
        epsilon = 0
        is_train = True
        
        # --------------------- Критический лосс ---------------------
        # Текущее Q: Q(s_t,a_t) = Q_Z(s_t, g(a_t))
        inverse_output = self.infer_hyper_action(observation, policy_output, self.actor)
        inverse_output.update({'state': policy_output['state'], 'action': inverse_output['Z']})
        current_critic_output = self.apply_critic(observation, inverse_output, self.critic)
        current_Q = current_critic_output['q']  # (B,)

        
        # Целевое Q
        next_policy_output = self.apply_policy(next_observation, self.actor_target, epsilon, False, is_train)
        target_critic_output = self.apply_critic(next_observation, next_policy_output, self.critic_target)
        next_Q = target_critic_output['q']  # (B,)

        with torch.no_grad():
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, True, is_train)
            inverse_output = self.infer_hyper_action(observation, temp_policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inverse_output['Z'], temp_policy_output['hyper_action']).mean()

        
        reward = user_feedback['reward'].view(-1) + self.eta * hyper_actor_loss


        
        target_Q = reward + self.gamma * (done_mask * next_Q).detach()
        
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        if do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        # actor loss
        
        if do_actor_update and self.actor_lr > 0:
            # potential loss
            self.potential_optimizer.zero_grad()
            
            # forward для potential_loss
            tmp_policy_output = self.apply_policy(observation, self.actor, 
                                                       epsilon, self.do_explore_in_train, is_train)
            potential_policy = self.apply_potential(observation, tmp_policy_output, self.potential)
            potential_data = self.apply_potential(observation, policy_output, self.potential)
            
            potential_loss = -potential_policy['f'].mean() + self.w * potential_data['f'].mean()
            potential_loss.backward()                 # backward (граф тут же освобождается)
            self.potential_optimizer.step()           # обновили потенциал


            self.actor_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, self.do_explore_in_train, is_train)
            critic_output = self.apply_critic(observation, temp_policy_output, self.critic)
            potential_policy_new = self.apply_potential(observation, temp_policy_output, self.potential)
            actor_loss = -critic_output['q'].mean() - potential_policy_new['f'].mean()
            # Optimize the actor 
            actor_loss.backward()
            self.actor_optimizer.step()

        # hyper actor loss
        if do_actor_update and self.hyper_actor_coef > 0:
            self.inverse_module_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, True, is_train)
            inverse_output = self.infer_hyper_action(observation, temp_policy_output, self.actor)
            hyper_actor_loss = self.hyper_actor_coef * F.mse_loss(inverse_output['Z'], temp_policy_output['hyper_action']).mean()
            # Optimize the actor 
            hyper_actor_loss.backward()
            self.inverse_module_optimizer.step()


        # pointwise supervised loss
        # behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)
        if do_actor_update and self.behavior_lr > 0:
            # (B, K)
            A = policy_output['effect_action']
            # (B, K)
            point_label = user_feedback['immediate_response'].view(self.batch_size, self.env.slate_size, self.env.response_dim)[:,:,0]

            self.actor_behavior_optimizer.zero_grad()
            temp_policy_output = self.apply_policy(observation, self.actor, 
                                                   epsilon, False, is_train)
            candidate_scores = torch.gather(temp_policy_output['all_preds'], 1, A-1)
            action_prob = torch.sigmoid(candidate_scores)
            behavior_loss = F.binary_cross_entropy(action_prob, point_label).mean()
        
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
            
        return critic_loss, actor_loss, hyper_actor_loss, behavior_loss
    
    def infer_hyper_action(self, observation, policy_output, actor):
        '''
        inverse function or pooling for A --> Z
        @input:
        - observation
        - policy_output
        - actor
        @output:
        - Z_cap
        '''
        # (B,K)
        A = policy_output['effect_action'] 
        # (B,K,item_dim)
        item_info = self.env.get_candidate_info({'item_id': A}, all_item=False) 
        # (B,K,item_enc_dim)
        item_enc, reg = actor.user_encoder.get_item_encoding(
                            A, {k[3:]: v for k,v in item_info.items() if k != 'item_id'}, 
                            self.batch_size)
        item_enc = item_enc.view(self.batch_size, actor.effect_action_dim, actor.enc_dim)
        # (B,K,hyper_action_dim)
        itemwise_inverse = self.inverse_module(item_enc).view(self.batch_size, actor.effect_action_dim, actor.action_dim)
        # (B, hyper_action_dim)
        recovered_Z = torch.mean(itemwise_inverse, dim = 1).view(self.batch_size, actor.action_dim)
        return {'Z': recovered_Z}

        
    def apply_critic(self, observation, policy_output, critic_model):
        # feed_dict = {"state_emb": policy_output["state_emb"], 
        #              "action_emb": policy_output["action_emb"]}
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['action']}
        critic_output = critic_model(feed_dict)
        return critic_output

    def apply_potential(self, observation, policy_output, potential_model):
        feed_dict = {'state': policy_output['state'],
                     'action': policy_output['action']}
        potential_output = potential_model(feed_dict)
        return {'f': potential_output['q']}

    def save(self):
        super().save()

    def load(self):
        super().load()
 