import torch
import torch.nn as nn

from src.general import BaseModel
from src.components import DNN


class KRMBUserResponse(BaseModel):
    '''
    KuaiRand Multi-Behavior user response model
    '''
    def __init__(
        self, 

        model_path,
        loss,
        l2_coef,
        
        user_latent_dim,
        item_latent_dim,
        enc_dim,
        attn_n_head,
        transformer_d_forward,
        transformer_n_layer,
        state_hidden_dims,
        scorer_hidden_dims,
        dropout_rate,

        reader_stats,

        device
            
        ):
        super().__init__(model_path, loss, l2_coef, device)

        self.user_latent_dim = user_latent_dim
        self.item_latent_dim = item_latent_dim
        self.enc_dim = enc_dim
        self.attn_n_head = attn_n_head
        self.scorer_hidden_dims = scorer_hidden_dims
        self.dropout_rate = dropout_rate
        self.bce_loss = nn.BCEWithLogitsLoss(reduction = 'none')
        self.state_dim = 3 * enc_dim

        stats = reader_stats

        self.reader_stats = reader_stats

        self.user_feature_dims = stats['user_feature_dims'] # {feature_name: dim}
        self.item_feature_dims = stats['item_feature_dims'] # {feature_name: dim}

        # user embedding
        self.uIDEmb = nn.Embedding(stats['n_user']+1, user_latent_dim)
        self.uFeatureEmb = {}
        for f,dim in self.user_feature_dims.items():
            embedding_module = nn.Linear(dim, user_latent_dim)
            self.add_module(f'UFEmb_{f}', embedding_module)
            self.uFeatureEmb[f] = embedding_module
            
        # item embedding
        self.iIDEmb = nn.Embedding(stats['n_item']+1, item_latent_dim)
        self.iFeatureEmb = {}
        for f,dim in self.item_feature_dims.items():
            embedding_module = nn.Linear(dim, item_latent_dim)
            self.add_module(f'IFEmb_{f}', embedding_module)
            self.iFeatureEmb[f] = embedding_module
        
        # feedback embedding
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.xtr_dim = 2*self.feedback_dim
        self.feedbackEncoder = nn.Linear(self.feedback_dim, enc_dim)
        self.set_behavior_hyper_weight(torch.ones(self.feedback_dim))
        
        # item embedding kernel encoder
        self.itemEmbNorm = nn.LayerNorm(item_latent_dim)
        self.userEmbNorm = nn.LayerNorm(user_latent_dim)
        self.itemFeatureKernel = nn.Linear(item_latent_dim, enc_dim)
        self.userFeatureKernel = nn.Linear(user_latent_dim, enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(enc_dim)
        
        # positional embedding
        self.max_len = stats['max_seq_len']
        self.posEmb = nn.Embedding(self.max_len, enc_dim)
        self.pos_emb_getter = torch.arange(self.max_len, dtype = torch.long)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))
        
        # sequence encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2*enc_dim, dim_feedforward = transformer_d_forward, 
                                                   nhead=attn_n_head, dropout = dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_n_layer)
        
        # DNN state encoder
        self.stateNorm = nn.LayerNorm(enc_dim)
        
        # DNN scorer
        self.scorer_hidden_dims = scorer_hidden_dims
        self.scorer = DNN(3*enc_dim, state_hidden_dims, self.feedback_dim * enc_dim, 
                          dropout_rate = dropout_rate, do_batch_norm = True)
        
        print("SIMULATOR layers:")
        print(f"Positional embeddings: {self.posEmb}")
        print(f"Transformer layer: {self.transformer}")
        print(f"Scorer layer: {self.scorer}")

        
    def to(self, device):
        new_self = super(KRMBUserResponse, self).to(device)
        # new_self.attn_mask = new_self.attn_mask.to(device)
        new_self.pos_emb_getter = new_self.pos_emb_getter.to(device)
        new_self.behavior_weight = new_self.behavior_weight.to(device)
        return new_self
        
    def set_behavior_hyper_weight(self, weight):
        self.behavior_weight = weight.view(-1)
        assert len(self.behavior_weight) == self.feedback_dim

    def get_forward(self, feed_dict: dict):
        '''
        This is used during simulator training
        When serving as a simulator, it calls encode_state() + get_pointwise_score()
        @input:
        - feed_dict: {
            'user_id': (B,)
            'uf_{feature_name}': (B,feature_dim), the user features
            'item_id': (B,), the target item
            'if_{feature_name}': (B,feature_dim), the target item features
            'history': (B,max_H)
            'history_if_{feature_name}': (B,max_H,feature_dim), the history item features
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,-1,n_feedback), 'reg': scalar}
        '''
        B = feed_dict['user_id'].shape[0]
        
        # target item
        # (B, -1, enc_dim)
        item_enc, item_reg = self.get_item_encoding(feed_dict['item_id'], 
                                          {k[3:]:v for k,v in feed_dict.items() if k[:3] == 'if_'}, B)
        # (B, -1, 1, enc_dim)
        item_enc = item_enc.view(B,-1,1,self.enc_dim)
        
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # (B, 1, 3*enc_dim)
        user_state = state_encoder_output['state'].view(B,1,3*self.enc_dim)
        # (B, -1, n_feedback), (B, -1, n_feedback)
        behavior_scores, point_scores = self.get_pointwise_scores(user_state, item_enc, B)
        

        # regularization terms
        reg = self.get_regularization(self.feedbackEncoder, 
                                      self.itemFeatureKernel, self.userFeatureKernel, 
                                      self.posEmb, self.transformer, self.scorer)
#         for v in self.uFeatureEmb.values():
#             reg += self.get_regularization(v)
#         for v in self.iFeatureEmb.values():
#             reg += self.get_regularization(v)
        reg = reg + state_encoder_output['reg'] + item_reg
        return {'preds': behavior_scores, 'state': user_state, 'reg': reg}
#         output_dict['reg'] = reg
#         return output_dict
    
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
        # user history
        # (B, max_H, enc_dim)
        history_enc, history_reg = self.get_item_encoding(feed_dict['history'], 
                                             {f:feed_dict[f'history_if_{f}'] for f in self.iFeatureEmb}, B)
        history_enc = history_enc.view(B, self.max_len, self.enc_dim)
        # (1, max_H, enc_dim)
        pos_emb = self.posEmb(self.pos_emb_getter).view(1,self.max_len,self.enc_dim)
        # (B, max_H, enc_dim)
        seq_enc_feat = self.encNorm(self.encDropout(history_enc + pos_emb))
        # (B, max_H, enc_dim)
        feedback_emb = self.get_response_embedding(feed_dict, B)
        # (B, max_H, 2*enc_dim)
        seq_enc = torch.cat((seq_enc_feat, feedback_emb), dim = -1)
        # (B, max_H, 2*enc_dim)
        output_seq = self.transformer(seq_enc, mask = self.attn_mask)
        # (B, 2*enc_dim)
        hist_enc = output_seq[:,-1,:].view(B,2*self.enc_dim)
        # user features
        # (B, enc_dim), scalar
        user_enc, user_reg = self.get_user_encoding(feed_dict['user_id'], 
                                          {k[3:]:v for k,v in feed_dict.items() if k[:3] == 'uf_'}, B)
        # (B, enc_dim)
        user_enc = self.encNorm(self.encDropout(user_enc)).view(B,self.enc_dim)
        # (B, 3*enc_dim)
        state = torch.cat([hist_enc,user_enc], 1)
        return {'output_seq': output_seq, 'state': state, 'reg': user_reg + history_reg}
    
    def get_user_encoding(self, user_ids, user_features, B):
        '''
        @input:
        - user_ids: (B,), encoded user id
        - user_features: {'uf_{feature_name}': (B, feature_dim)}
        '''
        # (B, 1, u_latent_dim)
        user_id_emb = self.uIDEmb(user_ids).view(B,1,self.user_latent_dim)
        # [(B, 1, u_latent_dim)] * n_user_feature
        user_feature_emb = [user_id_emb]
        for f,fEmbModule in self.uFeatureEmb.items():
            user_feature_emb.append(fEmbModule(user_features[f]).view(B,1,self.user_latent_dim))
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
        - item_ids: (B,) or (B,H), encoded item id
        - item_features: {'if_{feature_name}': (B,feature_dim) or (B,H,feature_dim)}
        '''
        # (B, 1, i_latent_dim) or (B, H, i_latent_dim)
        item_id_emb = self.iIDEmb(item_ids).view(B,-1,self.item_latent_dim)
        L = item_id_emb.shape[1]
        # [(B, 1, i_latent_dim)] * n_item_feature or [(B, H, i_latent_dim)] * n_item_feature
        item_feature_emb = [item_id_emb]
        for f,fEmbModule in self.iFeatureEmb.items():
            f_dim = self.item_feature_dims[f]
            item_feature_emb.append(fEmbModule(item_features[f].view(B,L,f_dim)).view(B,-1,self.item_latent_dim))
        # (B, 1, n_item_feature+1, i_latent_dim) or (B, H, n_item_feature+1, i_latent_dim)
        combined_item_emb = torch.cat(item_feature_emb, -1).view(B, L, -1, self.item_latent_dim)
        combined_item_emb = self.itemEmbNorm(combined_item_emb)
        # (B, 1, enc_dim) or (B, H, enc_dim)
        encoding = self.itemFeatureKernel(combined_item_emb).sum(2)
        encoding = encoding.view(B, -1, self.enc_dim)
        encoding = self.encNorm(encoding)
        # regularization
        reg = torch.mean(item_id_emb * item_id_emb)
        return encoding, reg
        
    def get_response_embedding(self, feed_dict, B):
        resp_list = []
        for f in self.feedback_types:
            # (B, max_H)
            resp = feed_dict[f'history_{f}'].view(B, self.max_len)
            resp_list.append(resp)
        # (B, max_H, n_feedback)
        combined_resp = torch.cat(resp_list, -1).view(B,self.max_len,self.feedback_dim)
        # (B, max_H, i_latent_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {...}
        - out_dict: {"preds":, "reg":}
        
        Loss terms implemented:
        - BCE
        """
        B = feed_dict['user_id'].shape[0]
        # (B, -1, n_feedback)
        preds = out_dict['preds'].view(B,-1,self.feedback_dim)
        # [(B, -1, 1)] * n_feedback
        targets = {f:feed_dict[f].view(B,-1).to(torch.float) for f in self.feedback_types}
        # (B, -1, n_feedback)
        loss_weight = feed_dict['loss_weight'].view(B,-1,self.feedback_dim)
        
        if self.loss_type == 'bce':
            behavior_loss = {}
            loss = 0
            for i,fb in enumerate(self.feedback_types):
                if self.behavior_weight[i] == 0:
                    continue
                Y = targets[fb].view(-1)
                P = preds[:,:,i].view(-1)
                W = loss_weight[:,:,i].view(-1)
                # (B*L,)
                point_loss = self.bce_loss(self.sigmoid(P), Y)
                behavior_loss[fb] = torch.mean(point_loss).item()
                point_loss = torch.mean(point_loss * W)
                point_loss = torch.mean(point_loss)
                loss = self.behavior_weight[i] * point_loss + loss
        else:
            raise NotImplemented
        out_dict['loss'] = loss + self.l2_coef * out_dict['reg']
        out_dict['behavior_loss'] = behavior_loss
        return out_dict
    
        
    def get_pointwise_scores(self, user_state, item_enc, B):
        '''
        Get user-item pointwise interaction scores
        @input:
        - user_state: (B, state_dim)
        - item_enc: (B, -1, 1, enc_dim) for batch-wise candidates or (1, -1, 1, enc_dim) for universal candidates
        - B: batch size
        @output:
        - behavior_scores: (B, -1, n_feedback)
        '''
        # scoring
        # (B, 1, n_feedback, enc_dim)
        behavior_attn = self.scorer(user_state).view(B,1,self.feedback_dim,self.enc_dim)
        # (B, 1, n_feedback, enc_dim)
        behavior_attn = self.stateNorm(behavior_attn)
        # (B, -1, n_feedback)
        point_scores = (behavior_attn * item_enc).mean(dim = -1).view(B,-1,self.feedback_dim)
        return point_scores, torch.mean(point_scores, dim = -1)
    
    