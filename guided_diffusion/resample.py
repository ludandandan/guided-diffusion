from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion) #返回一个均匀采样器，其权重全部被初始化为1
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):#这是一个父类
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective. 扩散过程中各个时间步的分布，旨在减小目标函数的方差

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged. 默认情况下，采样器执行无偏重要性采样，其中目标函数的均值保持不变
    However, subclasses may override sample() to change how the resampled 然而，子类可以重写sample方法，以改变重新采样项的重新加权方式，
    terms are reweighted, allowing for actual changes in the objective.从而允许实际改变目标函数
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step. 获取一个numpy数组，每个扩散步骤一个权重

        The weights needn't be normalized, but must be positive. 权重不必归一化，但必须是正数
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch. 为一个批次，重要性采样时间步

        :param batch_size: the number of timesteps. 时间步的数量
        :param device: the torch device to save to. 保存到的torch设备
        :return: a tuple (timesteps, weights): 一个元组（时间步，权重）
                 - timesteps: a tensor of timestep indices. 一个时间步索引的张量
                 - weights: a tensor of weights to scale the resulting losses. 一个张量，用于缩放结果损失的权重
        """
        w = self.weights() #获取权重（这个是采样权重）
        p = w / np.sum(w) #归一化，p大小是 [diffusion.num_timesteps]
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p) #根据权重进行采样，从0到len(p)-1中采样batch_size个数，每个数被采样的的概率为p
        indices = th.from_numpy(indices_np).long().to(device) #将indices_np转化为tensor
        weights_np = 1 / (len(p) * p[indices_np]) #计算权重，1/(len(p)*p[indices_np])，采样权重p越大，权重越小
        weights = th.from_numpy(weights_np).float().to(device) #将weights_np转化为tensor
        return indices, weights #返回indices和weights，大小都是[batch_size]


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps]) #初始化权重为1

    def weights(self):
        return self._weights #所有的权重都是1，即从0到timestep-1均匀采样


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model. 使用模型的损失更新重新加权

        Call this method from each rank with a batch of timesteps and the 从每个排名（进程）调用此方法，提供一批时间步和这些时间步对应的损失
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting. 这个方法将会执行同步，以确保所有排名（进程）保持完全相同的重新加权

        :param local_ts: an integer Tensor of timesteps. 一个整数张量，表示时间步
        :param local_losses: a 1D Tensor of losses. 一个1D张量，表示损失
        """
        batch_sizes = [ # 这里应该是都初始化为0
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size()) # dist.get_world_size()返回进程数。每个进程都会创建一个长度为1的tensor，值为0
        ]
        dist.all_gather(#从每个进程中收集其本地时间步数 local_ts 的长度信息，并将这些信息存储在 batch_sizes 列表中
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size. 将所有收集的批次填充到最大批次大小
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes) #找到最大的批量

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes] #创建一个长度为max_bs的tensor，值为0，device为local_ts.device，有多少个进程就创建多少个tensor
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes] #创建一个长度为max_bs的tensor，值为0，device为local_losses.device，有多少个进程就创建多少个tensor
        dist.all_gather(timestep_batches, local_ts)#每个进程收集其本地时间步 local_ts 的信息，并将这些信息存储在 timestep_batches 列表中
        dist.all_gather(loss_batches, local_losses)#每个进程收集其本地损失 local_losses 的信息，并将这些信息存储在 loss_batches 列表中
        timesteps = [ #timestep_batches放的是每个进程的时间步，batch_sizes放的是每个进程的时间步数
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ] #timesteps 是一个包含了从 timestep_batches 中提取的时间步数据的列表
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]] #losses 是一个包含了从 loss_batches 中提取的损失数据的列表
        self.update_with_all_losses(timesteps, losses) #调用update_with_all_losses方法，更新重新加权

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model. 使用模型的损失更新重新加权

        Sub-classes should override this method to update the reweighting
        using losses from the model. 子类应该重写此方法，以使用模型的损失来更新重新加权

        This method directly updates the reweighting without synchronizing这个方法直接更新重新加权，而不在工作进程之间进行同步。
        between workers. It is called by update_with_local_losses from all它由来自所有排名的具有相同参数的 update_with_local_losses 调用
        ranks with identical arguments. Thus, it should have deterministic因此，为了在工作进程之间保持状态一致，它应该具有确定性行为。
        behavior to maintain state across workers.

        :param ts: a list of int timesteps. 一个int时间步的列表
        :param losses: a list of float losses, one per timestep. 一个float损失的列表，每个时间步一个
        """


class LossSecondMomentResampler(LossAwareSampler): #这个类继承了LossAwareSampler，需要实现update_with_all_losses方法
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term #history_per_term是一个整数，表示每个时间步的历史损失数，要存10个历史损失
        self.uniform_prob = uniform_prob #uniform_prob是一个浮点数，表示均匀采样的概率（可能是）
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        ) #初始化一个二维数组，第一维是时间步，第二维是历史损失数（10），值为0
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int) #初始化一个一维数组，长度为时间步，值为0（以后里面会存已经存储的历史损失数）

    def weights(self):
        #训练初期，仍是随机采样步数
        if not self._warmed_up(): #如果所有时间步与历史损失数不相等（10），返回一个全为1的数组
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64) #权重全为1
        #训练后期，根据历史损失数计算权重（根据Lt的历史值计算pt）
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1)) # 计算平方平均并归一化，axis=-1表示最后一维，即对第二维求平均，原来是[diffusion.num_timesteps, history_per_term]，现在是[diffusion.num_timesteps
        weights /= np.sum(weights) #归一化
        weights *= 1 - self.uniform_prob #乘以1-uniform_prob=0.999
        weights += self.uniform_prob / len(weights) #加上uniform_prob/len(weights)=0.001/len(weights)，对权重进行平滑处理
        return weights #就是根据最近10个历史损失计算权重（权重近似等于历史loss的平方平均再开方再归一化）

    def update_with_all_losses(self, ts, losses): #ts是一个列表，包含了从 timestep_batches 中提取的时间步数据，losses 是一个包含了从 loss_batches 中提取的损失数据的列表
        for t, loss in zip(ts, losses): #当前第t个时间步的损失为loss
            if self._loss_counts[t] == self.history_per_term: #如果历史损失数等于history_per_term，即10
                # Shift out the oldest loss term. 将最旧的损失项移出
                self._loss_history[t, :-1] = self._loss_history[t, 1:] #将第t个时间步的历史损失向前移动一位（把第0位去掉，然后全部前移一位）
                self._loss_history[t, -1] = loss#将最新的损失放到最后一位
            else: #如果历史损失数不等于history_per_term，即小于10
                self._loss_history[t, self._loss_counts[t]] = loss #第t步的第_loss_counts[t]个历史损失放到_loss_history[t,self._loss_counts[t]]中
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all() #如果所有的时间步的历史损失数都等于history_per_term，返回True，否则返回False
