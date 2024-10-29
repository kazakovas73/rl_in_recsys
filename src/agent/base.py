import time
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm

from src.utils import LinearScheduler

class BaseRLAgent():
    '''
    RL Agent controls the overall learning algorithm:
    - objective functions for the policies and critics
    - design of reward function
    - how many steps to train
    - how to do exploration
    - loading and saving of models
    
    Main interfaces:
    - train
    '''
    
    def __init__(
        self, 

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

        env, 
        actor, 
        buffer,

        logger
    ):
        self.reward_func = reward_func
        self.n_iter = n_iter
        self.train_every_n_step = train_every_n_step
        self.start_policy_train_at_step = start_policy_train_at_step

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.elbow_epsilon = elbow_epsilon
        self.explore_rate = explore_rate
        self.do_explore_in_train = do_explore_in_train

        self.check_episode = check_episode
        self.save_episode = save_episode
        self.save_path = save_path
        
        self.actor_lr = actor_lr
        self.actor_decay = actor_decay
        self.batch_size = batch_size

        self.device = device
        self.logger = logger

        # components
        self.env = env
        self.actor = actor
        self.buffer = buffer
        
        # controller
        self.exploration_scheduler = LinearScheduler(
            int(sum(n_iter) * elbow_epsilon), 
            final_epsilon, 
            initial_p=initial_epsilon
        )
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=actor_lr, 
            weight_decay=actor_decay
        )

        # register modules that will be saved
        self.registered_models = [(self.actor, self.actor_optimizer, '_actor')]
        
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f" ")
    
    def train(self):
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        self.logger.info("Run procedures before training")
        self.action_before_train()
        t = time.time()
        start_time = t
        
        # training
        self.logger.info("Training:")
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
                self.logger.info(f"Episode step {i}, time diff {t_prime - t}, total time diff {t - start_time})")
                episode_report, train_report = self.get_report(smoothness = self.check_episode)
                log_str = f"step: {i} @ online episode: {episode_report} @ training: {train_report}\n"
                with open(self.save_path + ".report", 'a') as outfile:
                    outfile.write(log_str)
                self.logger.info(log_str)
                t = t_prime

            # save model and training info
            if i % self.save_episode == 0:
                self.save()
               
        self.action_after_train()
        
    
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
        
        for i in tqdm(range(self.start_policy_train_at_step)):
            do_explore = np.random.random() < self.explore_rate
            observation = self.run_episode_step(episode_iter, pre_epsilon, observation, 
                                                do_buffer_update, do_explore)
            prepare_step += 1
        print(f"Total {prepare_step} prepare steps")
        
    def setup_monitors(self):
        self.training_history = {'actor_loss': []}
        self.eval_history = {'avg_reward': [],
                             'reward_variance': [],
                             'avg_total_reward': [0.],
                             'max_total_reward': [0.],
                             'min_total_reward': [0.]}
        self.eval_history.update({f'{resp}_rate': [] for resp in self.env.response_types})
        self.current_sum_reward = torch.zeros(self.env.episode_batch_size).to(torch.float).to(self.device)
        
    
    def action_after_train(self):
        self.env.stop()
        
    def get_report(self, smoothness = 10):
        episode_report = self.env.get_report(smoothness)
        train_report = {k: np.mean(v[-smoothness:]) for k,v in self.training_history.items()}
        train_report.update({k: np.mean(v[-smoothness:]) for k,v in self.eval_history.items()})
        return episode_report, train_report

    def run_episode_step(self, *episode_args):
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
        episode_iter, epsilon, observation, do_buffer_update, do_explore = episode_args
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
    
    def get_reward(self, user_feedback):
        user_feedback['immediate_response_weight'] = self.env.response_weights
        R = self.reward_func(user_feedback).detach()
        return R
    
    def step_train(self):
        '''
        @process:
        '''
        observation, policy_output, user_feedback, done_mask, next_observation = self.buffer.sample(self.batch_size)
        
        loss_dict = self.get_loss(observation, policy_output, user_feedback, done_mask, next_observation)
        
        for k in loss_dict:
            if k in self.training_history:
                try:
                    self.training_history[k].append(loss_dict[k].item())
                except:
                    self.training_history[k].append(loss_dict[k])
    
    def get_loss(self, observation, policy_output, user_feedback, done_mask, next_observation):
        pass
    
    def test(self):
        pass
    
    def save(self):
        for model, opt, prefix in self.registered_models:
            torch.save(model.state_dict(), self.save_path + prefix)
            torch.save(opt.state_dict(), self.save_path + prefix + "_optimizer")
    
    def load(self):
        for model, opt, prefix in self.registered_models:
            model.load_state_dict(torch.load(self.save_path + prefix, map_location = self.device))
            opt.load_state_dict(torch.load(self.save_path + prefix + "_optimizer", map_location = self.device))
    