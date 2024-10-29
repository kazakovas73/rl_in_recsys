import torch.nn as nn
import torch
import numpy as np

from src.general import BaseModel
from src.policy.backbone import BackboneUserEncoder
from src.components import DNN
from src.utils import sample_categorical_action


class OneStagePolicy(BaseModel):
    def __init__(
        self,
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
        logger
    ):
        self.slate_size = env.slate_size
        # self.effect_action_dim = env.slate_size
        # self.hyper_action_dim = enc
        super().__init__(model_path, loss, l2_coef, device)
        self.display_name = "OneStagePolicy"

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

        self.enc_dim = self.user_encoder.enc_dim
        self.state_dim = self.user_encoder.state_dim
        self.action_dim = self.slate_size

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.dropout_rate = self.user_encoder.dropout_rate

    def to(self, device):
        new_self = super(OneStagePolicy, self).to(device)
        self.user_encoder.device = device
        self.user_encoder = self.user_encoder.to(device)
        return new_self

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'observation':{
                'user_profile':{
                    'user_id': (B,)
                    'uf_{feature_name}': (B,feature_dim), the user features}
                'user_history':{
                    'history': (B,max_H)
                    'history_if_{feature_name}': (B,max_H,feature_dim), the history item features}
            'candidates':{
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features}
            'epsilon': scalar, 
            'do_explore': boolean,
            'candidates': {
                'item_id': (B,L) or (1,L), the target item
                'item_{feature_name}': (B,L,feature_dim) or (1,L,feature_dim), the target item features},
            'action_dim': slate size K,
            'action': (B,K),
            'response': {
                'reward': (B,),
                'immediate_response': (B,K*n_feedback)},
            'is_train': boolean,
            'batch_wise': boolean
        }
        @output:
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'all_prob': (B,L),
            'action': (B,K),
            'reg': scalar}
        '''
        observation = feed_dict['observation']
        # observation --> user state
        state_dict = self.get_user_state(observation)
        # user state + candidates --> dict(state, prob, action, reg)
        out_dict = self.generate_action(state_dict, feed_dict)
        out_dict['state'] = state_dict['state']
        out_dict['reg'] = state_dict['reg'] + out_dict['reg']
        return out_dict

    def get_user_state(self, observation):
        feed_dict = {}
        feed_dict.update(observation['user_profile'])
        feed_dict.update(observation['user_history'])
        return self.user_encoder.get_forward(feed_dict)

    def get_loss_observation(self):
        return ['loss']

    def generate_action(self, state_dict, feed_dict):
        '''
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration
        * Categorical sampling: probabilistic exploration
        * Uniform sampling: random exploration

        This function will be called in the following places:
        * agent.run_episode_step() during online inference, corresponding input_dict:
            {'action': None, 'response': None, 'epsilon': >0, 'do_explore': True, 'is_train': False}
        * agent.step_train() during online training, correpsonding input_dict:
            {'action': tensor, 'response': see buffer.sample@output - user_response, 
             'epsilon': 0, 'do_explore': False, 'is_train': True}
        * agent.test() during test, corresponding input_dict:
            {'action': None, 'response': None, 'epsilon': 0, 'do_explore': False, 'is_train': False}

        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as self.get_forward@input - feed_dict
        @output:
        - out_dict: {'prob': (B, K), 
                     'all_prob': (B, L),
                     'action': (B, K), 
                     'reg': scalar}
        '''
        pass

    def get_loss(self, feed_dict, out_dict):
        '''
        @input:
        - feed_dict: same as get_forward@input-feed_dict
        - out_dict: {
            'state': (B,state_dim), 
            'prob': (B,K),
            'action': (B,K),
            'reg': scalar, 
            'immediate_response': (B,K),
            'reward': (B,)}
        @output
        - loss
        '''
        pass


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


class OneStageHyperPolicy_with_DotScore(OneStagePolicy):

    def __init__(
        self,
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

        policy_noise_var,
        policy_noise_clip,
        policy_do_effect_action_explore,
        policy_action_hidden,

        device,
        env,
        logger
    ):
        '''
        components:
        - user_encoder
        - hyper_action_layer
        - state_dim, enc_dim, action_dim
        '''
        self.noise_var = policy_noise_var
        self.noise_clip = policy_noise_clip
        self.do_effect_action_explore = policy_do_effect_action_explore
        super().__init__(
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
            logger
        )
        # action is the set of parameters of linear mapping [item_dim + 1, 1]
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

    def generate_action(self, state_dict, feed_dict):
        '''
        List generation provides three main types of exploration:
        * Greedy top-K: no exploration, set do_effect_action_explore=False in args or do_explore=False in feed_dict
        * Categorical sampling: probabilistic exploration, set do_effect_action_explore=True in args, 
                                set do_explore=True and epsilon < 1 in feed_dict
        * Uniform sampling : random exploration, set do_effect_action_explore=True in args,
                             set do_explore=True, epsilon > 0 in feed_dict
        * Gaussian sampling on hyper-action: set do_explore=True, epsilon < 1 in feed_dict
        * Uniform sampling on hyper-action: set do_explore=True, epsilon > 0 in feed_dict

        @input:
        - state_dict: {'state': (B, state_dim), ...}
        - feed_dict: same as OneStagePolicy.get_forward@input - feed_dict
        @output:
        - out_dict: {'preds': (B, K), 
                     'action': (B, hyper_action_dim), 
                     'indices': (B, K),
                     'hyper_action': (B, hyper_action_dim),
                     'effect_action': (B, K),
                     'all_preds': (B, L),
                     'reg': scalar}
        '''
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

    def get_score(self, hyper_action, candidate_item_enc, item_dim):
        '''
        Deterministic mapping from hyper-action to effect-action (rec list)
        '''
        # (B, L)
        scores = linear_scorer(hyper_action, candidate_item_enc, item_dim)
        return scores

    def forward(self, feed_dict: dict, return_prob=True):
        out_dict = self.get_forward(feed_dict)
        if return_prob:
            out_dict['all_probs'] = torch.softmax(out_dict['all_preds'], dim=1)
            out_dict['probs'] = torch.gather(
                out_dict['all_probs'], 1, out_dict['indices'])
        return out_dict
