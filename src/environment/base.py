from src.reader import *
from src.utils import wrap_batch


class BaseRLEnvironment():
    def __init__(
        self,
        max_step_per_episode,
        initial_temper,
        device,
        model_path,
        model,
        reader_stats,

        from_load=True
    ):

        self.device = device
        self.max_step_per_episode = max_step_per_episode
        self.initial_temper = initial_temper
        self.model_path = model_path

        if from_load:
            model.load_from_checkpoint(self.model_path, with_optimizer=False)
        self.model = model.to(device)

        self.reader_stats = reader_stats

    def reset(self, params):
        pass

    def step(self, action):
        pass

    def get_observation_from_batch(self, sample_batch):
        '''
        extract observation from the reader's sample batch
        @input:
        - sample_batch: {
            'user_id': (B,)
            'uf_{feature}': (B,F_dim(feature)), user features
            'history': (B,max_H)
            'history_length': (B,)
            'history_if_{feature}': (B, max_H * F_dim(feature))
            'history_{response}': (B, max_H)
            ... user ground truth feedbacks are not included as observation
        }
        @output:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H, feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B, )}}
        '''
        sample_batch = wrap_batch(sample_batch, device=self.device)
        profile = {'user_id': sample_batch['user_id']}
        for k, v in sample_batch.items():
            if 'uf_' in k:
                profile[k] = v
        history = {'history': sample_batch['history']}
        for k, v in sample_batch.items():
            if 'history_' in k:
                history[k] = v
        return {'user_profile': profile, 'user_history': history}
