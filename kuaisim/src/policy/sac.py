

import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.functional as F
from torch.distributions.normal import Normal

from src.components import DNN
from src.policy.backbone import BackboneUserEncoder
from src.general import BaseModel


LOG_STD_MAX = 2
LOG_STD_MIN = -5

def linear_scorer(action_emb, item_emb, item_dim):
    '''
    score = item_emb * weight + bias

    @input:
    - action_emb: (B, (i_dim+1))
    - item_emb: (B, L, i_dim) or (1, L, i_dim)
    @output:
    - score: (B, L)
    '''
    # scoring model parameters
    # (B, 1, i_dim)
    # * 2 / math.sqrt(self.item_dim)
    fc_weight = action_emb[:, :item_dim].view(-1, 1, item_dim)
    # (B, 1)
    fc_bias = action_emb[:, -1].view(-1, 1)

    # forward
    output = torch.sum(fc_weight * item_emb, dim=-1) + fc_bias
    # (B, L)
    return output


class Actor(BaseModel):
    def __init__(
        self, 

        # encoder
        model_path,
        loss,
        l2_coef,
        state_user_latent_dim,
        state_item_latent_dim,
        state_transformer_enc_dim,
        state_transformer_n_head,
        state_transformer_d_forward,
        state_transformer_n_layer,
        state_dropout_rate,
        device,
        env,
        logger,

        # network
        policy_noise_var,
        policy_noise_clip,
        policy_do_effect_action_explore,
        policy_action_hidden
        
    ):
        super().__init__(model_path, loss, l2_coef, device)
        self.display_name = "SAC actor"

        self.env = env
        
        
        self.user_encoder = BackboneUserEncoder(
            model_path,
            loss,
            l2_coef,
            state_user_latent_dim,
            state_item_latent_dim,
            state_transformer_enc_dim,
            state_transformer_n_head,
            state_transformer_d_forward,
            state_transformer_n_layer,
            state_dropout_rate,
            device,
            env.reader.get_statistics(),
            logger
        )

        self.noise_var = policy_noise_var
        self.noise_clip = policy_noise_clip
        self.do_effect_action_explore = policy_do_effect_action_explore

        self.slate_size = env.slate_size
        self.enc_dim = self.user_encoder.enc_dim
        self.state_dim = self.user_encoder.state_dim
        self.action_dim = self.slate_size
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dropout_rate = self.user_encoder.dropout_rate

        self.hyper_action_dim = self.enc_dim + 1
        self.action_dim = self.hyper_action_dim
        self.effect_action_dim = self.slate_size
        self.hyper_action_layer = DNN(
            in_dim=self.state_dim, 
            hidden_dims=policy_action_hidden, 
            out_dim=self.hyper_action_dim,
            dropout_rate=self.dropout_rate, 
            do_batch_norm=True
        )

        self.mu = nn.Linear(self.hyper_action_dim, self.hyper_action_dim)
        self.sigma = nn.Linear(self.hyper_action_dim, self.hyper_action_dim)
    
    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return self.user_encoder.get_forward(feed_dict)
    
    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores

    def sample_normal(self, user_state, reparameterize=True):

        B = user_state.shape[0]

        prob = self.hyper_action_layer(user_state).view(B, self.action_dim)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=1e-6, max=1)

        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.env.action_space).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+1e-6)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs
    
    def generate_action(self, state_dict, feed_dict):
        user_state = state_dict['state']
        candidates = feed_dict['candidates']
        # epsilon = feed_dict['epsilon']
        # do_explore = feed_dict['do_explore']
        # is_train = feed_dict['is_train']
        batch_wise = feed_dict['batch_wise']

        actions, _ = self.sample_normal(user_state, reparameterize=False)

        candidate_item_enc, reg = self.user_encoder.get_item_encoding(
            candidates['item_id'], {k[3:]: v for k, v in candidates.items() if k != 'item_id'}, B if batch_wise else 1)
        
        scores = self.get_score(actions, candidate_item_enc, self.enc_dim)

        _, indices = T.topk(scores, k=self.slate_size, dim=1)

        if batch_wise:
            action = torch.gather(candidates['item_id'], 1, indices).detach()  # (B, slate_size)
        else:
            # (B, slate_size)
            action = candidates['item_id'][indices].detach()

        action_scores = T.gather(scores, 1, indices).detach()

        reg += self.get_regularization(self)

        out_dict = {'preds': action_scores,
                    'action': actions,
                    'indices': indices,
                    'hyper_action': actions,
                    'effect_action': action,
                    'all_preds': scores,
                    'reg': reg}
        return out_dict

    def forward(self, feed_dict: dict, return_prob=True):
        observation = feed_dict['observation']
        state_dict = self.get_user_state(observation)
        out_dict = self.generate_action(state_dict, feed_dict)
        out_dict['state'] = state_dict['state']
        out_dict['reg'] = state_dict['reg'] + out_dict['reg']

        if return_prob:
            out_dict['all_probs'] = torch.softmax(out_dict['all_preds'], dim=1)
            out_dict['probs'] = torch.gather(
                out_dict['all_probs'], 1, out_dict['indices'])

        return out_dict