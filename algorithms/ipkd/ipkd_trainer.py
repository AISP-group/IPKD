import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from ipkd.utils.util import get_gard_norm, huber_loss, mse_loss
from ipkd.utils.valuenorm import ValueNorm
from ipkd.algorithms.utils.util import check


class IPKDTrainer:
    """
    Trainer class for IPKD to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (st_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 studentpolicy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.studentpolicy = studentpolicy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        self.ifdistill = args.ifdistill
        self.ifevalstudent = args.ifevalstudent
        self.ifofa = args.ifofa
        # Coefficient for corr_matrix regularization (IPM interaction penalty)
        self.corr_reg_coef = getattr(args, 'corr_reg_coef', 0.01)

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def ofa_loss(self, student_feature, teacher_feature, eps=1, temperature=1.):
        losses = [
            torch.sum(- (F.softmax(t / temperature, dim=1) ** eps) * torch.log(F.softmax(s / temperature, dim=1)),
                      dim=-1).mean()
            for s, t in zip(student_feature, teacher_feature)
        ]
        return torch.stack(losses).mean()

    def paws_loss(self, probs, targets, me_max=True):

        # probs[probs < 1e-4] *= 0  # numerical stability
        # targets[targets < 1e-4] *= 0  # numerical stability
        loss = torch.mean(torch.sum(torch.log(probs ** (-targets)), dim=-1))
        cross_loss = -torch.sum(targets * torch.log(probs + 1e-12), dim=1)
        probs_pre = torch.softmax(probs, dim=-1)
        probs_targets = torch.softmax(targets, dim=-1)
        loss_softmax = torch.mean(torch.sum(torch.log(probs_pre ** (-probs_targets)), dim=-1))
        print("归一化概率:", probs_pre)
        print("和:", probs_pre.sum())  # 应该接近1
        # cross_loss1 = nn.CrossEntropyLoss(probs,targets)
        # compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = torch.mean(probs_pre, dim=(0,1))
            rloss -= -torch.sum(avg_probs * torch.log(avg_probs + 1e-12), dim=0)
        return loss_softmax, rloss

    def l2_feature_loss(self, student_features,teacher_features):
        """Compute L2/MSE loss between intermediate features from student and teacher.

        The inputs are sequences (e.g., lists) of tensors where each pair corresponds
        to the same layer's output. Each pair loss is MSE over all elements, then
        averaged across layers.
        """
        pair_losses = [
            F.mse_loss(s, t, reduction='mean')
            for s, t in zip(student_features, teacher_features)
        ]
        if len(pair_losses) == 0:
            device = student_features[0].device if len(student_features) > 0 else 'cpu'
            return torch.tensor(0.0, device=device)
        return torch.stack(pair_losses).mean()

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, clifvalue, feature = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def student_update(self, sample):
        """
        Update student networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # 在不计算梯度的情况下，重新前向传播教师模型，得到教师模型的输出
        # 这样可以确保后续学生模型的反向传播不会影响到教师模型的参数和计算图
        with torch.no_grad():
            teacher_values, teacher_action_log_probs, teacher_dist_entropy, teacher_clifvalue, teacher_feature = self.policy.evaluate_actions(share_obs_batch,
                                                                                             obs_batch,
                                                                                             rnn_states_batch,
                                                                                             rnn_states_critic_batch,
                                                                                             actions_batch,
                                                                                             masks_batch,
                                                                                             available_actions_batch,
                                                                                             active_masks_batch)
        # Reshape to do in a single forward pass for all steps
        action_log_probs, distill_entropy, student_feature,corr_matrix = self.studentpolicy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            kl_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            kl_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # student update
        # 1. teacher-student action log_probs RMSE loss
        stu_mse = F.mse_loss(action_log_probs, teacher_action_log_probs, reduction='mean')
        student_mse = torch.sqrt(stu_mse)

        # 2. corr_matrix regularization: penalize deviation from identity
        # Prevents IPM from learning overly aggressive interaction patterns.
        # corr_matrix shape: [batch, n_agent, n_agent], diag is always 1.0 by construction.
        n_agent = corr_matrix.shape[-1]
        identity = torch.eye(n_agent, device=corr_matrix.device, dtype=corr_matrix.dtype)
        # L_reg = alpha * ||C - I||_F^2 (averaged over batch)
        corr_reg = self.corr_reg_coef * (torch.norm(corr_matrix - identity, p='fro', dim=(-2, -1)) ** 2).mean()

        if self.ifofa:
            # 3. feature-level distillation via MSE (method of feature-based OFA)
            feature_loss = self.l2_feature_loss(student_feature, teacher_feature)
            # total student loss
            distill_loss = student_mse + feature_loss + corr_reg - distill_entropy * self.entropy_coef
        else:
            distill_loss = student_mse + corr_reg - distill_entropy * self.entropy_coef

        self.studentpolicy.optimizer.zero_grad()
        distill_loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.studentpolicy.actor.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.studentpolicy.actor.parameters())

        self.studentpolicy.optimizer.step()

        return distill_loss, grad_norm, distill_entropy, corr_reg

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)


        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['distill_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['distill_entropy'] = 0
        train_info['student_grad_norm'] = 0
        train_info['corr_reg'] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, self.num_mini_batch)

            for sample in data_generator:
                # teacher model update
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights\
                    = self.ppo_update(sample)

                # student model update (now returns corr_reg instead of corr_matrix)
                distill_loss, student_grad_norm, distill_entropy, corr_reg = self.student_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['distill_loss'] += distill_loss.item()

                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['distill_entropy'] += distill_entropy.item()
                train_info['student_grad_norm'] += student_grad_norm.item()
                train_info['corr_reg'] += corr_reg.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.train()
        self.studentpolicy.train()

    def prep_rollout(self):
        self.studentpolicy.eval()


