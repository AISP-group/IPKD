import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from ipkd.algorithms.utils.util import check, init
from ipkd.algorithms.utils.transformer_act import discrete_autoregreesive_act
from ipkd.algorithms.utils.transformer_act import discrete_parallel_act
from ipkd.algorithms.utils.transformer_act import continuous_autoregreesive_act
from ipkd.algorithms.utils.transformer_act import continuous_parallel_act

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class InteractionPatternModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.L_param = nn.Parameter(torch.randn(n, n))

    def forward(self, batch_size):
        # 只取下三角
        L = torch.tril(self.L_param)
        diag_indices = torch.arange(self.n)
        L[diag_indices, diag_indices] = torch.nn.functional.softplus(L[diag_indices, diag_indices])
        Sigma = L @ L.T
        D = torch.sqrt(torch.diag(Sigma))
        corr = Sigma / (D[:, None] * D[None, :])
        # 扩展到batch
        corr = corr.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n, n]
        return corr

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

class CustomActor(nn.Module):

    def __init__(self, obs_dim, n_embd, action_dim, n_head, n_agent):
        super(CustomActor, self).__init__()

        self.ln1 = nn.LayerNorm(obs_dim)
        self.fc1 = nn.Linear(obs_dim, n_embd)
        self.act1 = nn.GELU()
        self.addfc1 = nn.Linear(n_embd, n_embd)
        self.addact1 = nn.GELU()
        # self.proj1 = nn.Linear(n_embd, action_dim)
        self.ln2 = nn.LayerNorm(n_embd)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.act2 = nn.GELU()
        self.addfc2 = nn.Linear(n_embd, n_embd)
        self.addact2 = nn.GELU()
        # self.proj2 = nn.Linear(n_embd, action_dim)
        self.ln3 = nn.LayerNorm(n_embd)
        self.fc3 = nn.Linear(n_embd, action_dim)

    def forward(self, x):
        features = []
        x = self.ln1(x)
        x_cat = self.fc1(x)
        x = self.act1(x_cat)
        x = self.addfc1(x)
        x = self.addact1(x)
        x = x_cat + self.ln2(x)
        # y1 = self.proj1(x)
        y1 = x
        features.append(y1)
        x_catcat = self.fc2(x)
        x = self.act2(x_catcat)
        x = self.addfc2(x)
        x = self.addact2(x)
        x = x_catcat+self.ln3(x)
        y2 = x
        features.append(y2)

        x = self.fc3(x)

        # features_out = []
        # x = self.ln1(x)
        # x_cat = self.fc1(x)
        # x = self.act1(x_cat)
        # x = self.addfc1(x)
        # x = self.addact1(x)
        # x_feature1 = x_cat + self.ln2(x)
        #
        # x_catcat = self.fc2(x_feature1)
        # x = self.act2(x_catcat)
        # x = self.addfc2(x)
        # x = self.addact2(x)
        # x_feature2 = x_catcat + self.ln3(x)
        #
        # features = torch.cat([x_feature1, x_feature2], dim=-1)
        # features = features.unsqueeze(1)
        # feature_cross = self.sam(features)
        # half = feature_cross.shape[-1] // 2
        # # feature_1 = x[..., :feature_cross.shape[-1] // 2]  # 取最后一维前一半
        # # feature_2 = x[..., feature_cross.shape[-1] // 2:]  # 取最后一维后一半
        # feature_1, feature_2 = torch.split(feature_cross, half, dim=-1)
        # feature_1 = feature_1.squeeze(1)  # [1280, N]
        # feature_2 = feature_2.squeeze(1)  # [1280, N]
        # y1 = self.proj1(feature_1)
        # y2 = self.proj2(feature_2)
        # features_out.append(y1)
        # features_out.append(y2)
        # x = self.fc3(x_feature2)

        return x, features
class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x1, x2, requires_feat=False):

        x_cat = torch.cat([x1, x2], dim=-1)
        x = self.ln1(x_cat + self.attn1(x_cat, x_cat, x_cat))
        # y = x + obsrep_enc
        if requires_feat:
            return x,y
        else:
            return y

class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = x

        return rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_block = n_block

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.share_actor:
            self.mlp = CustomActor(obs_dim, n_embd, action_dim, n_head, n_agent)
        else:
            self.mlp = nn.ModuleList()
            for n in range(n_agent):
                # actor = nn.Sequential(nn.LayerNorm(obs_dim),
                #                       init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                #                       init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                #                       init_(nn.Linear(n_embd, action_dim)))
                actor = CustomActor(obs_dim, n_embd, action_dim, n_head, n_agent)
                self.mlp.append(actor)
                self.sam = EncodeBlock(n_embd * 2, n_head, n_agent)
                self.proj1 = nn.Linear(n_embd, action_dim)
                self.proj2 = nn.Linear(n_embd, action_dim)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        clifvalue = 0
        if self.share_actor:
            logit,feature = self.mlp(obs)
        else:
            logit = []
            first_feature = []
            second_feature = []
            features = []
            for n in range(len(self.mlp)):
                logit_n, feature_n = self.mlp[n](obs[:, n, :])
                logit.append(logit_n)
                first_feature.append(feature_n[0])
                second_feature.append(feature_n[1])
            logit = torch.stack(logit, dim=1)
            first_feature = torch.stack(first_feature, dim=1)
            second_feature = torch.stack(second_feature, dim=1)
            feature_cat = torch.cat([first_feature, second_feature], dim=-1)
            feature_cross = self.sam(feature_cat)

            half = feature_cross.shape[-1] // 2
            feature_1, feature_2 = torch.split(feature_cross, half, dim=-1)
            y1 = self.proj1(feature_1)
            y2 = self.proj2(feature_2)
            features.append(y1)
            features.append(y2)

        return logit, clifvalue, features



class StudentNetwork(nn.Module):

    def __init__(self, state_dim, obs_dim, obswithcorr_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(StudentNetwork, self).__init__()

        self.n_agent = n_agent
        self.ipm_net = InteractionPatternModel(n_agent)
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state used
        self.state_dim = state_dim

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obswithcorr_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        # ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        obs_rep = self.encoder(state, obs)

        # add Interaction Pattern Model to incorporate inter-agent correlation into obs
        corr_matrix = self.ipm_net(batch_size)  # [batch_size, num_agents, num_agents]
        corr_matrix = check(corr_matrix).to(**self.tpdv)
        obs_with_corr = torch.cat([obs, corr_matrix], dim=-1)

        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy, clifvalue, feature = discrete_parallel_act(self.decoder, obs_rep, obs_with_corr, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy, clifvalue, feature = continuous_parallel_act(self.decoder, obs_rep, obs_with_corr, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, entropy, feature, corr_matrix

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        # ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        obs_rep = self.encoder(state, obs)

        # add Interaction Pattern Model to incorporate inter-agent correlation into obs
        corr_matrix = self.ipm_net(batch_size)  # [batch_size, num_agents, num_agents]
        # corr_row = torch.stack([corr_matrix[:, i, :] for i in range(self.num_agents)],dim=1)  # [batch, num_agents, num_agents]
        corr_matrix = check(corr_matrix).to(**self.tpdv)
        obs_with_corr = torch.cat([obs, corr_matrix], dim=-1)

        if self.action_type == "Discrete":
            output_action, output_action_log, clifvalue, feature = discrete_autoregreesive_act(self.decoder, obs_rep, obs_with_corr, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log, clifvalue, feature = continuous_autoregreesive_act(self.decoder, obs_rep, obs_with_corr, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log




