from collections import OrderedDict

import numpy as np
import torch

from resnet_server import ResnetServerRoll


# Algorithm 1, line 7; line 10 (Eq. (2)) is ResnetServerRoll.combine
class ResnetServerImportance(ResnetServerRoll):
    criterion = 'hope'

    def __init__(self, global_model, rate, dataset_ref, cfg_id):
        super().__init__(global_model, rate, dataset_ref, cfg_id)
        self.refresh_every = self.cfg.get('rank_refresh', 0)
        self.warmup = self.cfg.get('rank_warmup', 0)
        self.rank_idx = None
        self.alloc = self.cfg.get('alloc', 'uniform')
        self.width_maps = None
        # Algorithm 1, lines 1-3
        if self.warmup > 0:
            self.rank_idx = self._identity_rankings()
        else:
            self.refresh_rankings()

    def _identity_rankings(self):
        return {k: torch.arange(v.size(0))
                for k, v in self.global_parameters.items()
                if k.split('.')[-1] == 'weight' and v.dim() > 1
                and ('conv1' in k or 'conv2' in k)}

    def _ensure_width_maps(self):
        if self.width_maps is not None or self.alloc == 'uniform':
            return
        from layer_alloc import width_map
        rates = sorted({float(r) for r in list(self.rate) + list(self.model_rate)})
        self.width_maps = {r: width_map(self.global_model, r, self.alloc,
                                        seed=self.cfg.get('init_seed', 0))
                           for r in rates}
        print('layer_alloc[{}]: width profiles for rates {}'.format(self.alloc, rates),
              flush=True)

    def refresh_rankings(self):
        from hope_rank import channel_rankings
        self.rank_idx = channel_rankings(self.global_model, self.criterion,
                                         seed=self.cfg.get('init_seed', 0))

    def step(self, local_parameters, param_idx, user_idx):
        super().step(local_parameters, param_idx, user_idx)
        if self.warmup and self.rounds == self.warmup:
            self.refresh_rankings()
        elif self.refresh_every and self.rounds % self.refresh_every == 0:
            if not self.warmup or self.rounds > self.warmup:
                self.refresh_rankings()

    def split_model(self, user_idx):
        if self.warmup and self.rounds < self.warmup:
            return super().split_model(user_idx)

        cfg = self.cfg
        idx_i = [None for _ in range(len(user_idx))]
        idx = [OrderedDict() for _ in range(len(user_idx))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split('.')[-1]
            for m in range(len(user_idx)):
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    if parameter_type == 'weight':
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if 'conv1' in k or 'conv2' in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx_i_m = idx_i[m]
                                scaler_rate = self.model_rate[user_idx[m]] / cfg['global_model_rate']
                                if self.alloc == 'uniform':
                                    # K_{m,i} = ceil(beta_m K_i), Sec. 2
                                    local_output_size = int(np.ceil(output_size * scaler_rate))
                                else:
                                    self._ensure_width_maps()
                                    wm = self.width_maps[float(self.model_rate[user_idx[m]])]
                                    local_output_size = wm.get(k, int(np.ceil(output_size * scaler_rate)))
                                # Algorithm 1, line 7
                                model_idx = self.rank_idx[k].to(v.device)
                                output_idx_i_m = model_idx[:local_output_size]
                                idx_i[m] = output_idx_i_m
                            elif 'shortcut' in k:
                                input_idx_i_m = idx[m][k.replace('shortcut', 'conv1')][1]
                                output_idx_i_m = idx_i[m]
                            elif 'linear' in k:
                                input_idx_i_m = idx_i[m]
                                output_idx_i_m = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError('Not valid k')
                            idx[m][k] = (output_idx_i_m, input_idx_i_m)
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                    else:
                        input_size = v.size(0)
                        if 'linear' in k:
                            input_idx_i_m = torch.arange(input_size, device=v.device)
                            idx[m][k] = input_idx_i_m
                        else:
                            input_idx_i_m = idx_i[m]
                            idx[m][k] = input_idx_i_m
                else:
                    pass

        return idx


# FedHOPE, Eq. (3)
class ResnetServerHope(ResnetServerImportance):
    criterion = 'hope'


# Table 1: BN scale
class ResnetServerSlim(ResnetServerImportance):
    criterion = 'slim'


# Table 1: l1
class ResnetServerL1(ResnetServerImportance):
    criterion = 'l1'


# Table 1: random-fixed, the random order of the Corollary
class ResnetServerRandFix(ResnetServerImportance):
    criterion = 'randfix'
