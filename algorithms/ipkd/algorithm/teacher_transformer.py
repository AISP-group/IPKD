import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from ipkd.algorithms.utils.util import check, init
from ipkd.algorithms.utils.transformer_block import discrete_autoregreesive_act
from ipkd.algorithms.utils.transformer_block import discrete_parallel_act
from ipkd.algorithms.utils.transformer_block import continuous_autoregreesive_act
from ipkd.algorithms.utils.transformer_block import continuous_parallel_act


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


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

    def forward(self, x, obsrep_enc, staterep_enc, requires_feat=False):

        x_prior = x[:, 1:, :]
        x_cat = torch.cat([staterep_enc, x_prior], dim=1)
        x = self.ln1(x_cat + self.attn1(x_cat, x_cat, x_cat))
        y = x + obsrep_enc
        if requires_feat:
            return x,y
        else:
            return y


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, device="cuda:0"):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        self.device = device
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.  state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        clifvalue1 = 0
        clifvalue2 = 0

        state_embeddings = self.state_encoder(state)

        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            i = 0
            state_embeddings_i = state_embeddings[:, i:i + 1, :]
            x = torch.cat([state_embeddings_i, obs_embeddings], dim=1)

        rep = self.blocks(self.ln(x))
        rep2 = rep[:, :1, :]
        rep1 = rep[:, 1:, :]
        v_loc = self.head(rep1)

        return v_loc, rep1, rep2, clifvalue1, clifvalue2

class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, device="cuda:0",
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent
        self.device = device

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.featureproj1 = nn.ModuleList([init_(nn.Linear(n_embd, action_dim)) for _ in range(n_agent)])
            self.featureproj2 = nn.ModuleList([init_(nn.Linear(n_embd, action_dim)) for _ in range(n_agent)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))
            self.critichead = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, 1)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, state_rep, obs, gt_action):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        feature = []
        outs1 = []
        for i in range(self.n_agent):
            out = self.featureproj1[i](obs_rep[:, i, :])  # [batch, out_features]
            outs1.append(out.unsqueeze(1))      # [batch, 1, out_features]
        # 合并成 [batch, 3, out_features]
        feature1 = torch.cat(outs1, dim=1)
        feature.append(feature1)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep, state_rep)
            clifvalue = 0

            # x is the output of outer transformer block, will be used to put in ActorHead and CriticHead
            logit = self.head(x)
            v_val = self.critichead(x)
            outs2 = []
            for i in range(self.n_agent):
                out = self.featureproj2[i](x[:, i, :])  # [batch, out_features]
                outs2.append(out.unsqueeze(1))  # [batch, 1, out_features]
            # 合并成 [batch, 3, out_features]
            feature2 = torch.cat(outs2, dim=1)
            feature.append(feature2)
        return logit, v_val, clifvalue, feature

class TeacherTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(TeacherTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state used
        self.state_dim = state_dim
        print("block_num: ", n_block)
        print("state_dim: ", state_dim)

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, self.device)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent, self.device,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)

        self.to(self.device)

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
        v_loc, obs_rep,state_rep, rewardvalue, statevalue = self.encoder(state, obs)

        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy, v_value, clifvalue, feature = discrete_parallel_act(self.decoder, obs_rep, state_rep, obs, action, batch_size,
                                                                   self.n_agent, self.action_dim, self.tpdv,
                                                                   available_actions)
        else:
            action_log, entropy, v_value, clifvalue, feature = continuous_parallel_act(self.decoder, obs_rep, state_rep, obs, action, batch_size,
                                                                     self.n_agent, self.action_dim, self.tpdv)

        clifvalue = statevalue

        # use the critic head output v_value replace original v_loc
        #return action_log, v_loc, entropy, clifvalue
        return action_log, v_value, entropy, clifvalue, feature

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        # ori_shape = np.shape(obs)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep, state_rep, rewardvalue, stateval = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log, v_value, clifvalue, feature = discrete_autoregreesive_act(self.decoder, obs_rep, state_rep, obs,
                                                                                      batch_size,
                                                                                      self.n_agent, self.action_dim,
                                                                                      self.tpdv,
                                                                                      available_actions, deterministic)
        else:
            output_action, output_action_log, v_value, clifvalue, feature = continuous_autoregreesive_act(self.decoder, obs_rep, state_rep, obs,
                                                                                        batch_size,
                                                                                        self.n_agent, self.action_dim,
                                                                                        self.tpdv,
                                                                                        deterministic)
        clifvalue = stateval

        # use the critic head output v_value replace original v_loc
        #return output_action, output_action_log, v_loc, clifvalue
        return output_action, output_action_log, v_value, clifvalue

    def get_values(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        # ori_shape = np.shape(state)
        # state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)


        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep, state_rep, rewardvalue, stateval = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log, v_value, clifvalue, feature = discrete_autoregreesive_act(self.decoder, obs_rep, state_rep, obs,
                                                                                      batch_size,
                                                                                      self.n_agent, self.action_dim,
                                                                                      self.tpdv,
                                                                                      available_actions, deterministic)
        else:
            output_action, output_action_log, v_value, clifvalue, feature = continuous_autoregreesive_act(self.decoder, obs_rep, state_rep, obs,
                                                                                        batch_size,
                                                                                        self.n_agent, self.action_dim,
                                                                                        self.tpdv,
                                                                                        deterministic)
        # v_totandsem1 = rewardvalue
        # v_totandsem2 = semanticval
        # v_tot = v_tot+v_totandsem1+v_totandsem2

        return v_value


