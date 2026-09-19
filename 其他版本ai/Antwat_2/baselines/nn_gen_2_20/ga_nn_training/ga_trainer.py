import numpy as np


class TwoStageScoringNetwork:
    """两阶段评分网络：状态编码 + 动作评分"""

    def __init__(self, model_path=None):
        self.state_dim = 10150
        self.state_embed_dim = 512
        self.action_dim = 21

        # Encoder
        self.enc_w1 = np.zeros((self.state_dim, 2048), dtype=np.float32)
        self.enc_b1 = np.zeros(2048, dtype=np.float32)
        self.enc_w2 = np.zeros((2048, 1024), dtype=np.float32)
        self.enc_b2 = np.zeros(1024, dtype=np.float32)
        self.enc_w3 = np.zeros((1024, self.state_embed_dim), dtype=np.float32)
        self.enc_b3 = np.zeros(self.state_embed_dim, dtype=np.float32)

        # Scorer
        self.scorer_w1 = np.zeros((self.state_embed_dim + self.action_dim, 256), dtype=np.float32)
        self.scorer_b1 = np.zeros(256, dtype=np.float32)
        self.scorer_w2 = np.zeros((256, 128), dtype=np.float32)
        self.scorer_b2 = np.zeros(128, dtype=np.float32)
        self.scorer_w_out = np.zeros((128, 1), dtype=np.float32)
        self.scorer_b_out = np.zeros(1, dtype=np.float32)

        # Score normalization
        self.score_mean = 0.0
        self.score_std = 1.0
        self.max_abs_score = 2094.42

        if model_path is not None:
            self.load(model_path)

    def encode_state(self, state_vec):
        x = state_vec @ self.enc_w1 + self.enc_b1
        x = np.maximum(x, 0)
        x = x @ self.enc_w2 + self.enc_b2
        x = np.maximum(x, 0)
        x = x @ self.enc_w3 + self.enc_b3
        return x

    def score_action(self, state_embed, action_vec):
        x = np.concatenate([state_embed, action_vec])
        x = x @ self.scorer_w1 + self.scorer_b1
        x = np.maximum(x, 0)
        x = x @ self.scorer_w2 + self.scorer_b2
        x = np.maximum(x, 0)
        score = (x @ self.scorer_w_out + self.scorer_b_out)[0]

        normalized_score = (score - self.score_mean) / self.score_std
        normalized_score = normalized_score * self.max_abs_score

        return normalized_score

    def save(self, path):
        data = {
            'enc_w1': self.enc_w1,
            'enc_b1': self.enc_b1,
            'enc_w2': self.enc_w2,
            'enc_b2': self.enc_b2,
            'enc_w3': self.enc_w3,
            'enc_b3': self.enc_b3,
            'scorer_w1': self.scorer_w1,
            'scorer_b1': self.scorer_b1,
            'scorer_w2': self.scorer_w2,
            'scorer_b2': self.scorer_b2,
            'scorer_w_out': self.scorer_w_out,
            'scorer_b_out': self.scorer_b_out,
            'score_mean': np.float32(self.score_mean),
            'score_std': np.float32(self.score_std),
        }
        np.savez(path, **data)

    def load(self, path):
        data = np.load(path)
        self.enc_w1 = data['enc_w1']
        self.enc_b1 = data['enc_b1']
        self.enc_w2 = data['enc_w2']
        self.enc_b2 = data['enc_b2']
        self.enc_w3 = data['enc_w3']
        self.enc_b3 = data['enc_b3']

        self.scorer_w1 = data['scorer_w1']
        self.scorer_b1 = data['scorer_b1']
        self.scorer_w2 = data['scorer_w2']
        self.scorer_b2 = data['scorer_b2']
        self.scorer_w_out = data['scorer_w_out']
        self.scorer_b_out = data['scorer_b_out']

        self.score_mean = float(data['score_mean'])
        self.score_std = float(data['score_std'])


class TwoStageGenome:
    """两阶段神经网络基因组"""

    STATE_DIM = 10150
    ACTION_DIM = 21
    STATE_EMBED_DIM = 512

    def __init__(self):
        self.fitness = 0.0
        self.generation = 0
        self.weights = {}

    def save(self, path):
        np.savez(path, **self.weights)

    def load(self, path):
        data = np.load(path)
        self.weights = {key: data[key] for key in data.files}

    def create_network(self):
        network = TwoStageScoringNetwork.__new__(TwoStageScoringNetwork)
        network.state_dim = self.STATE_DIM
        network.state_embed_dim = self.STATE_EMBED_DIM
        network.action_dim = self.ACTION_DIM

        # Encoder
        network.enc_w1 = self.weights['enc_w1']
        network.enc_b1 = self.weights['enc_b1']
        network.enc_w2 = self.weights['enc_w2']
        network.enc_b2 = self.weights['enc_b2']
        network.enc_w3 = self.weights['enc_w3']
        network.enc_b3 = self.weights['enc_b3']

        # Scorer
        network.scorer_w1 = self.weights['scorer_w1']
        network.scorer_b1 = self.weights['scorer_b1']
        network.scorer_w2 = self.weights['scorer_w2']
        network.scorer_b2 = self.weights['scorer_b2']
        network.scorer_w_out = self.weights['scorer_w_out']
        network.scorer_b_out = self.weights['scorer_b_out']

        # Score normalization
        network.score_mean = float(self.weights['score_mean'])
        network.score_std = float(self.weights['score_std'])
        network.max_abs_score = 132.64

        return network
