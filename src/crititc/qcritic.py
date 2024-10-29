import torch.nn as nn
import torch

from src.components import DNN
from src.utils import get_regularization


class QCritic(nn.Module):

    def __init__(self, critic_hidden_dims, critic_dropout_rate, policy, logger):
        super().__init__()
        self.state_dim = policy.state_dim
        self.action_dim = policy.action_dim
        self.net = DNN(
            in_dim=self.state_dim + self.action_dim, 
            hidden_dims=critic_hidden_dims, 
            out_dim=1,
            dropout_rate=critic_dropout_rate, 
            do_batch_norm=True
        )
        logger.info("QCritic layers:")
        logger.info(f"NET: {self.net}")

    def forward(self, feed_dict):
        '''
        @input:
        - feed_dict: {'state': (B, state_dim), 'action': (B, action_dim)}
        '''
        state_emb = feed_dict['state'].view(-1, self.state_dim)
        action_emb = feed_dict['action'].view(-1, self.action_dim)
        Q = self.net(torch.cat((state_emb, action_emb), dim=-1)).view(-1)
        reg = get_regularization(self.net)
        return {'q': Q, 'reg': reg}
