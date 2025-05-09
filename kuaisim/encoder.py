from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn

from src.general import BaseModel
from src.components import DNN


class Encoder(BaseModel):
    '''
    KuaiRand Multi-Behavior user response model
    '''

    def __init__(
        self,
        model_path, loss, l2_coef,
        state_user_latent_dim,
        state_item_latent_dim,
        state_transformer_enc_dim,
        state_transformer_n_head,
        state_transformer_d_forward,
        state_transformer_n_layer,
        state_dropout_rate,
        device,
        reader_stats,
        logger
    ):
        super().__init__(model_path, loss, l2_coef, device)
        self.user_latent_dim = state_user_latent_dim
        self.item_latent_dim = state_item_latent_dim
        self.enc_dim = state_transformer_enc_dim
        self.state_dim = 3 * self.enc_dim
        self.attn_n_head = state_transformer_n_head
        self.dropout_rate = state_dropout_rate

        stats = reader_stats
        self.logger = logger

        # {feature_name: dim}
        self.user_feature_dims = stats['user_feature_dims']
        # {feature_name: dim}
        self.item_feature_dims = stats['item_feature_dims']

        self.logger.info("POLICY model layers:")

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user']+1, state_user_latent_dim)
        self.uFeatureEmb = {}
        for f, dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, state_user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
        self.logger.info(f"- USER embeddings layer: {self.uIDEmb}")

        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item']+1, state_item_latent_dim)
        self.iFeatureEmb = {}
        for f, dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, state_item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        self.logger.info(f"- ITEM embeddings layer: {self.iIDEmb}")

        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.feedbackEncoder = nn.Linear(
            self.feedback_dim, state_transformer_enc_dim)

        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(state_item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(state_user_latent_dim)
        self.itemFeatureKernel = nn.Linear(
            state_item_latent_dim, state_transformer_enc_dim)
        self.userFeatureKernel = nn.Linear(
            state_user_latent_dim, state_transformer_enc_dim)
        self.encDropout = nn.Dropout(state_dropout_rate)
        self.encNorm = nn.LayerNorm(state_transformer_enc_dim)

        self.max_len = stats['max_seq_len']
        # positional embedding
        self.posEmb = nn.Embedding(self.max_len, state_transformer_enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype=torch.long)
        self.attn_mask = ~torch.tril(torch.ones(
            (self.max_len, self.max_len), dtype=torch.bool))
        self.logger.info(f"- POSITIONAL embeddings layer: {self.posEmb}")

        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*state_transformer_enc_dim,
                                                   dim_feedforward=state_transformer_d_forward,
                                                   nhead=state_transformer_n_head,
                                                   dropout=state_dropout_rate,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=state_transformer_n_layer)
        self.logger.info(f"- TRANSFORMER layer: {self.transformer}")

    def to(self, device):
        new_self = super(Encoder, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        return new_self

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
        }
        @output:
        - out_dict: {'state': (B, state_dim), 
                    'reg': scalar}
        '''
        B = feed_dict['user_id'].shape[0]
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # regularization terms
        reg = self.get_regularization(self.feedbackEncoder,
                                      self.itemFeatureKernel, self.userFeatureKernel,
                                      self.posEmb, self.transformer)
        reg = reg + state_encoder_output['reg']

        return {'state': state_encoder_output['state'],
                'reg': reg}

    def encode_state(self, feed_dict, B):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
            ... (irrelevant input)
        }
        - B: batch size
        @output:
        - out_dict:{
            'out_seq': (B,max_H,2*enc_dim)
            'state': (B,n_feedback*enc_dim)
            'reg': scalar
        }
        '''
        # user history item encodings (B, max_H, enc_dim)
        history_enc, history_reg = self.get_item_encoding(feed_dict['history'],
                                                          {f: feed_dict[f'history_if_{f}'] for f in self.iFeatureEmb}, B)
        history_enc = history_enc.view(B, self.max_len, self.enc_dim)

        # positional encoding (1, max_H, enc_dim)
        pos_emb = self.posEmb(self.pos_emb_getter).view(
            1, self.max_len, self.enc_dim)

        # feedback embedding (B, max_H, enc_dim)
        feedback_emb = self.get_response_embedding(
            {f: feed_dict[f'history_{f}'] for f in self.feedback_types}, B)

        # sequence item encoding (B, max_H, enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        # (B, max_H, 2*enc_dim)
        seq_enc = torch.cat((seq_enc_feat, feedback_emb), dim=-1)

        # transformer output (B, max_H, 2*enc_dim)
        output_seq = self.transformer(seq_enc, mask=self.attn_mask)

        # user history encoding (B, 2*enc_dim)
        hist_enc = output_seq[:, -1, :].view(B, 2*self.enc_dim)

        # static user profile features
        # (B, enc_dim), scalar
        user_enc, user_reg = self.get_user_encoding(feed_dict['user_id'],
                                                    {k[3:]: v for k, v in feed_dict.items() if k[:3] == 'uf_'}, B)
        # (B, enc_dim)
        user_enc = self.encNorm(self.encDropout(
            user_enc)).view(B, self.enc_dim)

        # user state (B, 3*enc_dim) combines user history and user profile features
        state = torch.cat([hist_enc, user_enc], 1)
        # (B, enc_dim)
#         state = self.stateNorm(self.finalStateLayer(state))
        return {'output_seq': output_seq, 'state': state, 'reg': user_reg + history_reg}

    def get_user_encoding(self, user_ids, user_features, B):
        '''
        @input:
        - user_ids: (B,)
        - user_features: {'uf_{feature_name}': (B, feature_dim)}
        @output:
        - encoding: (B, enc_dim)
        - reg: scalar
        '''
        # (B, 1, u_latent_dim)
        user_id_emb = self.uIDEmb(user_ids).view(B, 1, self.user_latent_dim)
        # [(B, 1, u_latent_dim)] * n_user_feature
        user_feature_emb = [user_id_emb]
        for f, fEmbModule in self.uFeatureEmb.items():
            user_feature_emb.append(fEmbModule(
                user_features[f]).view(B, 1, self.user_latent_dim))
        # (B, n_user_feature+1, u_latent_dim)
        combined_user_emb = torch.cat(user_feature_emb, 1)
        combined_user_emb = self.userEmbNorm(combined_user_emb)
        # (B, enc_dim)
        encoding = self.userFeatureKernel(combined_user_emb).sum(1)
        # regularization
        reg = torch.mean(user_id_emb * user_id_emb)
        return encoding, reg

    def get_item_encoding(self, item_ids, item_features, B):
        '''
        @input:
        - item_ids: (B,) or (B,L)
        - item_features: {'{feature_name}': (B,feature_dim) or (B,L,feature_dim)}
        @output:
        - encoding: (B, 1, enc_dim) or (B, L, enc_dim)
        - reg: scalar
        '''
        # (B, 1, i_latent_dim) or (B, L, i_latent_dim)
        item_id_emb = self.iIDEmb(item_ids).view(B, -1, self.item_latent_dim)
        L = item_id_emb.shape[1]
        # [(B, 1, i_latent_dim)] * n_item_feature or [(B, L, i_latent_dim)] * n_item_feature
        item_feature_emb = [item_id_emb]
        for f, fEmbModule in self.iFeatureEmb.items():
            f_dim = self.item_feature_dims[f]
            item_feature_emb.append(fEmbModule(item_features[f].view(
                B, L, f_dim)).view(B, -1, self.item_latent_dim))
        # (B, 1, n_item_feature+1, i_latent_dim) or (B, L, n_item_feature+1, i_latent_dim)
        combined_item_emb = torch.cat(
            item_feature_emb, -1).view(B, L, -1, self.item_latent_dim)
        combined_item_emb = self.itemEmbNorm(combined_item_emb)
        # (B, 1, enc_dim) or (B, L, enc_dim)
        encoding = self.itemFeatureKernel(combined_item_emb).sum(2)
        encoding = self.encNorm(encoding.view(B, -1, self.enc_dim))
        # regularization
        reg = torch.mean(item_id_emb * item_id_emb)
        return encoding, reg

    def get_response_embedding(self, resp_dict, B):
        '''
        @input:
        - resp_dict: {'{response}': (B, max_H)}
        @output:
        - resp_emb: (B, max_H, enc_dim)
        '''
        resp_list = []
        for f in self.feedback_types:
            # (B, max_H)
            resp = resp_dict[f].view(B, self.max_len)
            resp_list.append(resp)
        # (B, max_H, n_feedback)
        combined_resp = torch.cat(
            resp_list, -1).view(B, self.max_len, self.feedback_dim)
        # (B, max_H, enc_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb
