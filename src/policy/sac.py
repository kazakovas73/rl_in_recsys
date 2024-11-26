

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, ):
        super().__init__()


        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    # def forward(self, feed_dict: dict, return_prob=True):
        
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     mean = self.fc_mean(x)
    #     log_std = self.fc_logstd(x)
    #     log_std = torch.tanh(log_std)
    #     log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

    #     return mean, log_std
    
    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict['all_probs'] = torch.softmax(out_dict['all_preds'], dim=1)
            out_dict['probs'] = torch.gather(
                out_dict['all_probs'], 1, out_dict['indices'])
        return out_dict

    # def get_action(self, x):
    #     mean, log_std = self(x)
    #     std = log_std.exp()
    #     normal = torch.distributions.Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
    #     log_prob = log_prob.sum(1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean
    
    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return self.user_encoder.get_forward(feed_dict)
    
    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict['state']  # (B, state_dim)
        candidates = feed_dict['candidates']
        epsilon = feed_dict['epsilon']
        do_explore = feed_dict['do_explore']
        is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']

        B = user_state.shape[0]
        # epsilon probability for uniform sampling under exploration
        do_uniform = np.random.random() < epsilon
        # (B, hyper_action_dim)
        hyper_action_raw = self.hyper_action_layer(
            user_state).view(B, self.action_dim)
#         print('hyper_action_raw:', hyper_action_raw.shape)

        # (B, hyper_action_dim), hyper action exploration
        if do_explore:
            if do_uniform:
                hyper_action = torch.clamp(torch.rand_like(hyper_action_raw)*self.noise_var,
                                           -self.noise_clip, self.noise_clip)
            else:
                hyper_action = hyper_action_raw + torch.clamp(torch.rand_like(hyper_action_raw)*self.noise_var,
                                                              -self.noise_clip, self.noise_clip)
        else:
            hyper_action = hyper_action_raw

        # (B, L, enc_dim) if batch_wise candidates, otherwise (1,L,enc_dim)
        candidate_item_enc, reg = self.user_encoder.get_item_encoding(candidates['item_id'],
                                                                      {k[3:]: v for k, v in candidates.items(
                                                                      ) if k != 'item_id'},
                                                                      B if batch_wise else 1)
#         print('candidate_item_enc:', candidate_item_enc.shape)
        # (B, L)
        scores = self.get_score(hyper_action, candidate_item_enc, self.enc_dim)
#         print('scores:', scores.shape)

        # effect action exploration in both training and inference
        if self.do_effect_action_explore and do_explore:
            if do_uniform:
                # categorical sampling
                action, indices = sample_categorical_action(P, candidates['item_id'],
                                                            self.slate_size, with_replacement=False,
                                                            batch_wise=batch_wise, return_idx=True)
            else:
                # uniform sampling happens only in inference time
                action, indices = sample_categorical_action(torch.ones_like(P), candidates['item_id'],
                                                            self.slate_size, with_replacement=False,
                                                            batch_wise=batch_wise, return_idx=True)
        else:
            # top-k selection
            _, indices = torch.topk(scores, k=self.slate_size, dim=1)
            if batch_wise:
                action = torch.gather(
                    candidates['item_id'], 1, indices).detach()  # (B, slate_size)
            else:
                # (B, slate_size)
                action = candidates['item_id'][indices].detach()
#         print('action:', action.shape)
#         print(action)
#         input()
        action_scores = torch.gather(scores, 1, indices).detach()

        reg += self.get_regularization(self.hyper_action_layer)

        out_dict = {'preds': action_scores,
                    'action': hyper_action,
                    'indices': indices,
                    'hyper_action': hyper_action,
                    'effect_action': action,
                    'all_preds': scores,
                    'reg': reg}
        return out_dict

    def get_forward(self, feed_dict: dict):
        observation = feed_dict['observation']
        # observation --> user state
        state_dict = self.get_user_state(observation)
        # user state + candidates --> dict(state, prob, action, reg)
        out_dict = self.generate_action(state_dict, feed_dict)
        out_dict['state'] = state_dict['state']
        out_dict['reg'] = state_dict['reg'] + out_dict['reg']
        return out_dict
