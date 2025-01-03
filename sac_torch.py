import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


def linear_scorer(action_emb, item_emb, item_dim):
    fc_weight = action_emb[:, :item_dim].view(-1, 1, item_dim)
    fc_bias = action_emb[:, -1].view(-1, 1)
    output = T.sum(fc_weight * item_emb, dim=-1) + fc_bias
    return output


class Agent():
    def __init__(self, encoder=None, buffer=None, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, gamma=0.99, action_dim=2, item_dim=2, slate_size=1, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        
        self.encoder = encoder
        self.gamma = gamma
        self.tau = tau
        self.memory = buffer
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.item_dim = item_dim
        self.slate_size = slate_size

        self.actor = ActorNetwork(alpha, input_dims, action_dim=action_dim,
                    name='actor', max_action=env.action_space)
        self.critic_1 = CriticNetwork(beta, input_dims, action_dim=action_dim,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, action_dim=action_dim,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return self.encoder.get_forward(feed_dict)
    
    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores
    
    def get_regularization(self, *modules):
        reg = 0
        for m in modules:
            for p in m.parameters():
                reg += T.mean(p * p)
        return reg

    def choose_action(self, feed_dict):

        observation = feed_dict['observation']
        state_dict = self.get_user_state(observation)

        user_state = state_dict['state']
        candidates = feed_dict['candidates']
        # epsilon = feed_dict['epsilon']
        # do_explore = feed_dict['do_explore']
        # is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']

        B = user_state.shape[0]

        actions, _ = self.actor.sample_normal(user_state, reparameterize=False)

        candidate_item_enc, reg = self.encoder.get_item_encoding(
            candidates['item_id'], {k[3:]: v for k, v in candidates.items() if k != 'item_id'}, B if batch_wise else 1)

        scores = self.get_score(actions, candidate_item_enc, self.item_dim)

        _, indices = T.topk(scores, k=self.slate_size, dim=1)

        action = candidates['item_id'][indices].detach()
        action_scores = T.gather(scores, 1, indices).detach()

        reg += self.get_regularization(self.actor)

        out_dict = {
            'state': state_dict['state'],
            'preds': action_scores,
            'action': actions,
            'indices': indices,
            'effect_action': action,
            'all_preds': scores,
            'reg': reg + state_dict['reg']
        }

        return out_dict

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

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

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self, env):
        # if self.memory.mem_cntr < self.batch_size:
        #     return

        # state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        observation, policy_output, user_feedback, done_mask, next_observation = self.memory.sample(self.batch_size)

        reward = user_feedback['reward']
        done = done_mask
        state_ = next_observation
        state = observation
        action = policy_output['action']

        # reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        # done = T.tensor(done).to(self.actor.device)
        # state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        # state = T.tensor(state, dtype=T.float).to(self.actor.device)
        # action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(policy_output['state']).view(-1)

        input_dict = {
                'observation': next_observation, 
                'candidates': env.get_candidate_info(next_observation), 
                # 'epsilon': epsilon, 
                # 'do_explore': do_explore, 
                # 'is_train': is_train, 
                'batch_wise': False
            }
        next_policy_output = self.choose_action(input_dict)
        
        value_ = self.target_value(next_policy_output['state']).view(-1)

        value_[done] = 0.0

        # value = self.value(state).view(-1)
        # value_ = self.target_value(state_).view(-1)
        # value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(policy_output['state'], reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(policy_output['state'], actions)
        q2_new_policy = self.critic_2.forward(policy_output['state'], actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        # log_probs = log_probs.view(-1)
        # q1_new_policy = self.critic_1.forward(state, actions)
        # q2_new_policy = self.critic_2.forward(state, actions)
        # critic_value = T.min(q1_new_policy, q2_new_policy)
        # critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(policy_output['state'], reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(policy_output['state'], actions)
        q2_new_policy = self.critic_2.forward(policy_output['state'], actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        # actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        # log_probs = log_probs.view(-1)
        # q1_new_policy = self.critic_1.forward(state, actions)
        # q2_new_policy = self.critic_2.forward(state, actions)
        # critic_value = T.min(q1_new_policy, q2_new_policy)
        # critic_value = critic_value.view(-1)
        
        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(policy_output['state'], action).view(-1)
        q2_old_policy = self.critic_2.forward(policy_output['state'], action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
