import numpy as np
import torch as T

class ReplayBuffer():
    def __init__(self, max_size, device, state_dim, action_dim):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # self.state_memory = np.zeros((self.mem_size, *input_shape))
        # self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        # self.action_memory = np.zeros((self.mem_size, n_actions))
        # self.reward_memory = np.zeros(self.mem_size)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        profile = {k: v[batch] for k, v in self.state_memory["user_profile"].items()}
        history = {k: v[batch] for k, v in self.state_memory["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}

        profile = {k: v[batch] for k, v in self.next_state_memory["user_profile"].items()}
        history = {k: v[batch] for k, v in self.next_state_memory["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}

        actions = {
            "state": self.action_memory["state"][batch],
            "action": self.action_memory["action"][batch],
            # "prob": self.action_memory["prob"][batch]
        }
        rewards = self.reward_memory[batch]
        # immediate_response = self.immediate_response[batch]
        dones = self.terminal_memory[batch]

        return observation, actions, rewards, next_observation, dones
    
    def reset(self, env):
        observation = env.create_observation_buffer(self.mem_size)
        next_observation = env.create_observation_buffer(self.mem_size)

        policy_output = {
            'state': T.zeros(self.mem_size, self.state_dim).to(T.float).to(self.device),
            'action': T.zeros(self.mem_size, self.action_dim).to(T.long).to(self.device),
            # 'prob': T.zeros(self.mem_size, env.slate_size).to(T.float).to(self.device)
        }
        reward = T.zeros(self.mem_size).to(T.float).to(self.device)
        done = T.zeros(self.mem_size).to(T.bool).to(self.device)
        im_response = T.zeros(self.mem_size, env.response_dim * env.slate_size).to(T.float).to(self.device)

        self.state_memory = observation
        self.next_state_memory = next_observation
        self.action_memory = policy_output
        self.reward_memory = reward
        self.immediate_response = im_response
        self.terminal_memory = done


class BaseBuffer():
    '''
    The general buffer
    '''

    def __init__(self, buffer_size, device, state_dim, action_dim):
        self.buffer_size = buffer_size
        super().__init__()
        self.device = device
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

    def reset(self, env):
        '''
        @output:
        - buffer: {'observation': {'user_profile': {'user_id': (L,), 
                                                    'uf_{feature_name}': (L, feature_dim)}, 
                                   'user_history': {'history': (L, max_H), 
                                                    'history_if_{feature_name}': (L, max_H * feature_dim), 
                                                    'history_{response}': (L, max_H), 
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim), 
                                     'action': (L, action_dim), 
                                     'prob': (L, slate_size)}, 
                   'next_observation': same format as @output-buffer['observation'], 
                   'done_mask': (L,),
                   'response': {'reward': (L,), 
                                'immediate_response':, (L, slate_size * response_dim)}}
        '''
        observation = env.create_observation_buffer(self.buffer_size)
        next_observation = env.create_observation_buffer(self.buffer_size)
        policy_output = {
            'state': T.zeros(self.buffer_size, self.state_dim).to(T.float).to(self.device),
            'action': T.zeros(self.buffer_size, self.action_dim).to(T.float).to(self.device),
            # 'prob': T.zeros(self.buffer_size, env.slate_size).to(T.float).to(self.device)
        }
        reward = T.zeros(self.buffer_size).to(T.float).to(self.device)
        done = T.zeros(self.buffer_size).to(T.bool).to(self.device)
        im_response = T.zeros(self.buffer_size, env.response_dim * env.slate_size)\
            .to(T.float).to(self.device)
        self.buffer = {'observation': observation,
                       'policy_output': policy_output,
                       'user_response': {'reward': reward, 'immediate_response': im_response},
                       'done_mask': done,
                       'next_observation': next_observation}
        return self.buffer

    def sample(self, batch_size):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, user_response, done_mask, next_observation)

        Buffer: see reset@output
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'state': (B, state_dim), 
                          'action': (B, slate_size), 
                          'prob': (B, slate_size)}, 
        - user_feedback: {'reward': (B,), 
                          'immediate_response':, (B, slate_size * response_dim)}}
        - done_mask: (B,),
        - next_observation: same format as @output - observation, 
        '''
        # get indices
        indices = np.random.randint(
            0, self.current_buffer_size, size=batch_size)
        # observation
        profile = {k: v[indices]
                   for k, v in self.buffer["observation"]["user_profile"].items()}
        history = {k: v[indices]
                   for k, v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        # next observation
        profile = {k: v[indices]
                   for k, v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k: v[indices]
                   for k, v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        # policy output
        policy_output = {"state": self.buffer["policy_output"]["state"][indices],
                         "action": self.buffer["policy_output"]["action"][indices],
                        #  "prob": self.buffer["policy_output"]["prob"][indices]
                         }
        # user response
        user_response = {"reward": self.buffer["user_response"]["reward"][indices],
                         "immediate_response": self.buffer["user_response"]["immediate_response"][indices]}
        # done mask
        done_mask = self.buffer["done_mask"][indices]
        return observation, policy_output, user_response, done_mask, next_observation

    def update(self, observation, policy_output, user_feedback, next_observation):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'user_state': (B, state_dim), 
                          'prob': (B, action_dim),
                          'action': (B, action_dim)}
        - user_feedback: {'done': (B,), 
                          'immdiate_response':, (B, action_dim * feedback_dim), 
                          'reward': (B,)}
        - next_observation: same format as update_buffer@input-observation
        '''
        # get buffer indices to update
        B = len(user_feedback['reward'])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = T.tensor(indices).to(T.long).to(self.device)

        # update buffer - observation
        for k, v in observation['user_profile'].items():
            self.buffer['observation']['user_profile'][k][indices] = v
        for k, v in observation['user_history'].items():
            self.buffer['observation']['user_history'][k][indices] = v
        # update buffer - next observation
        for k, v in next_observation['user_profile'].items():
            self.buffer['next_observation']['user_profile'][k][indices] = v
        for k, v in next_observation['user_history'].items():
            self.buffer['next_observation']['user_history'][k][indices] = v
        # update buffer - policy output
        self.buffer['policy_output']['state'][indices] = policy_output['state']
        self.buffer['policy_output']['action'][indices] = policy_output['action']
        # self.buffer['policy_output']['prob'][indices] = policy_output['prob']
        # update buffer - user response
        self.buffer['user_response']['immediate_response'][indices] = user_feedback['immediate_response'].view(
            B, -1)
        self.buffer['user_response']['reward'][indices] = user_feedback['reward']
        # update buffer - done
        self.buffer['done_mask'][indices] = user_feedback['done']

        # buffer pointer
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
