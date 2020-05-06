#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List, Dict
from torch.autograd import Variable
from itertools import product, permutations, combinations_with_replacement, chain


class UtilsConf(NamedTuple):
    emb_size: int
    spatial_size: int  # number of entities


class UtilSharedConf(NamedTuple):
    num_utils: int  # number of utils sharing weights
    connected_list: List[str]  # list of utils (outside of the group) that are connected to the group


class Unary(nn.Module):
    def __init__(self, embed_size):
        """
            Captures local entity information
        :param embed_size:  the embedding dimension
        """
        super(Unary, self).__init__()
        self.embed = nn.Conv1d(embed_size, embed_size, 1)
        self.feature_reduce = nn.Conv1d(embed_size, 1, 1)

    def forward(self, X):
        X = X.transpose(1, 2)

        X_embed = self.embed(X)

        X_nl_embed = F.dropout(F.relu(X_embed))
        X_poten = self.feature_reduce(X_nl_embed)
        return X_poten.squeeze(1)


class Pairwise(nn.Module):
    def __init__(self, embed_x_size, x_spatial_dim=None, embed_y_size=None, y_spatial_dim=None):
        """
            Captures interaction between utilities or entities of the same utility
        :param embed_x_size: the embedding dimension of the first utility
        :param x_spatial_dim: the spatial dimension of the first utility for batch norm and weighted marginalization
        :param embed_y_size: the embedding dimension of the second utility (none for self-interactions)
        :param y_spatial_dim: the spatial dimension of the second utility for batch norm and weighted marginalization
        """

        super(Pairwise, self).__init__()
        embed_y_size = embed_y_size if embed_y_size is not None else embed_x_size
        self.y_spatial_dim = y_spatial_dim if y_spatial_dim is not None else x_spatial_dim

        self.embed_size = max(embed_x_size, embed_y_size)
        self.x_spatial_dim = x_spatial_dim

        self.embed_X = nn.Conv1d(embed_x_size, self.embed_size, 1)
        self.embed_Y = nn.Conv1d(embed_y_size, self.embed_size, 1)
        if x_spatial_dim is not None:
            self.normalize_S = nn.BatchNorm1d(self.x_spatial_dim * self.y_spatial_dim)

            self.margin_X = nn.Conv1d(self.y_spatial_dim, 1, 1)
            self.margin_Y = nn.Conv1d(self.x_spatial_dim, 1, 1)

    def forward(self, X, Y=None):

        X_t = X.transpose(1, 2)
        Y_t = Y.transpose(1, 2) if Y is not None else X_t

        X_embed = self.embed_X(X_t)
        Y_embed = self.embed_Y(Y_t)

        X_norm = F.normalize(X_embed)
        Y_norm = F.normalize(Y_embed)

        S = X_norm.transpose(1, 2).bmm(Y_norm)
        if self.x_spatial_dim is not None:
            S = self.normalize_S(S.view(-1, self.x_spatial_dim * self.y_spatial_dim)) \
                .view(-1, self.x_spatial_dim, self.y_spatial_dim)

            X_poten = self.margin_X(S.transpose(1, 2)).transpose(1, 2).squeeze(2)
            Y_poten = self.margin_Y(S).transpose(1, 2).squeeze(2)
        else:
            X_poten = S.mean(dim=2, keepdim=False)
            Y_poten = S.mean(dim=1, keepdim=False)

        if Y is None:
            return X_poten
        else:
            return X_poten, Y_poten


class Atten(nn.Module):
    def __init__(self, utils_conf: Dict[str, UtilsConf], sharing_factor_weights=None, prior_flag=False,
                 size_force=False, pairwise_flag=True, unary_flag=True, self_flag=True):
        """
            The class performs an attention on a given list of utilities representation.
        :param utils_conf: configuration for each utility (embedding size, spatial size)
        :param sharing_factor_weights: high-order factors to share weights (i.e. similar utilities).
         in visual-dialog history interaction. the format should be [(idx, number of utils, connected_to utils)...]
        :param prior_flag: is prior factor provided
        :param sizes: the spatial simension (used for batch-norm and weighted marginalization)
        :param size_force: force spatial size with adaptive avg pooling.
        :param pairwise_flag: use pairwise interaction between utilities
        :param unary_flag: use local information
        :param self_flag: use self interactions between utilitie's entities
        """
        super(Atten, self).__init__()

        self.utils_conf = utils_conf

        self.prior_flag = prior_flag

        self.n_utils = len(utils_conf)

        self.spatial_pool = nn.ModuleDict()

        self.un_models = nn.ModuleDict()

        self.self_flag = self_flag
        self.pairwise_flag = pairwise_flag
        self.unary_flag = unary_flag
        self.size_force = size_force

        self.sharing_factor_weights = sharing_factor_weights if sharing_factor_weights is not None else {}
        self.sharing_factors_set = set([shared_util_key for shared_util_key in self.sharing_factor_weights])

        for util_name, util_conf in utils_conf.items():
            self.un_models[util_name] = Unary(util_conf.emb_size)
            if self.size_force:
                self.spatial_pool[util_name] = nn.AdaptiveAvgPool1d(util_conf.spatial_size)

        self.pp_models = nn.ModuleDict()
        for ((util_a_name, util_a_conf), (util_b_name, util_b_conf)) \
                in combinations_with_replacement(utils_conf.items(), 2):
            if util_a_name == util_b_name:
                self.pp_models[util_a_name] = Pairwise(util_a_conf.emb_size, util_a_conf.spatial_size)
            else:
                if pairwise_flag:
                    for util_name, util_shared_conf in self.sharing_factor_weights.items():
                        if util_name == util_a_name and util_b_name not in set(util_shared_conf.connected_list) \
                                or util_b_name == util_name and util_a_name not in set(util_shared_conf.connected_list):
                            continue
                    self.pp_models[f"({util_a_name}, {util_b_name})"] = Pairwise(util_a_conf.emb_size,
                                                                                 util_a_conf.spatial_size,
                                                                                 util_b_conf.emb_size,
                                                                                 util_b_conf.spatial_size)

        self.reduce_potentials = nn.ModuleDict()
        self.num_of_potentials = dict()

        self.default_num_of_potentials = 0

        if self.self_flag:
            self.default_num_of_potentials += 1
        if self.unary_flag:
            self.default_num_of_potentials += 1
        if self.prior_flag:
            self.default_num_of_potentials += 1
        for util_name in self.utils_conf.keys():
            self.num_of_potentials[util_name] = self.default_num_of_potentials

        '''
         All other utilities
        '''
        if pairwise_flag:
            for util_name, util_shared_conf in self.sharing_factor_weights.items():
                for c_u_name in util_shared_conf.connected_list:
                    self.num_of_potentials[c_u_name] += util_shared_conf.num_utils
                    self.num_of_potentials[util_name] += 1
            for util_name in self.num_of_potentials.keys():
                if util_name not in self.sharing_factors_set:
                    self.num_of_potentials[util_name] += (self.n_utils - 1) - len(self.sharing_factor_weights)

        for util_name in self.utils_conf.keys():
            self.reduce_potentials[util_name] = nn.Conv1d(self.num_of_potentials[util_name], 1, 1, bias=False)

    def forward(self, utils: dict, b_size, priors=None):
        assert self.n_utils == len(utils)
        assert (priors is None and not self.prior_flag) \
               or (priors is not None
                   and self.prior_flag
                   and len(priors) == self.n_utils)
        # b_size = utils[0].size(0)
        util_poten = dict()
        attention = list()
        if self.size_force:
            for util_name, util_shared_conf in self.sharing_factor_weights.items():
                if util_name not in self.spatial_pool.keys():
                    continue
                else:
                    high_util = utils[util_name]
                    high_util = high_util.view(util_shared_conf.num_utils * b_size, high_util.size(2), high_util.size(3))
                    high_util = high_util.transpose(1, 2)
                    utils[util_name] = self.spatial_pool[util_name](high_util).transpose(1, 2)

            for util_name in self.utils_conf.keys():
                if util_name in self.sharing_factors_set \
                        or util_name not in self.spatial_pool.keys():
                    continue
                utils[util_name] = utils[util_name].transpose(1, 2)
                utils[util_name] = self.spatial_pool[util_name](utils[util_name]).transpose(1, 2)
                if self.prior_flag and priors[util_name] is not None:
                    priors[util_name] = self.spatial_pool[util_name](priors[util_name].unsqueeze(1)).squeeze(1)

        for util_name, util_shared_conf in self.sharing_factor_weights.items():
            # i.e. High-Order utility
            if self.unary_flag:
                util_poten.setdefault(util_name, []).append(self.un_models[util_name](utils[util_name]))

            if self.self_flag:
                util_poten.setdefault(util_name, []).append(self.pp_models[util_name](utils[util_name]))

            if self.pairwise_flag:
                for c_u_name in util_shared_conf.connected_list:
                    other_util = utils[c_u_name]
                    expanded_util = other_util.unsqueeze(1).expand(b_size,
                                                                   util_shared_conf.num_utils,
                                                                   other_util.size(1),
                                                                   other_util.size(2)).contiguous().view(
                        b_size * util_shared_conf.num_utils,
                        other_util.size(1),
                        other_util.size(2))

                    try:
                        model_key = f"({util_name}, {c_u_name})"
                        poten_ij, poten_ji = self.pp_models[model_key](utils[util_name], expanded_util)
                    except KeyError:
                        try:
                            model_key = f"({c_u_name}, {util_name})"
                            poten_ji, poten_ij = self.pp_models[model_key](expanded_util, utils[util_name])
                        except Exception as e:
                            print(e)
                            raise

                    util_poten[util_name].append(poten_ij)
                    util_poten.setdefault(c_u_name, []).append(poten_ji.view(b_size, util_shared_conf.num_utils,
                                                                             poten_ji.size(1)))

        # local
        for util_name in self.utils_conf.keys():
            if util_name in self.sharing_factors_set:
                continue
            if self.unary_flag:
                util_poten.setdefault(util_name, []).append(self.un_models[util_name](utils[util_name]))
            if self.self_flag:
                util_poten.setdefault(util_name, []).append(self.pp_models[util_name](utils[util_name]))

        # joint
        if self.pairwise_flag:
            for (util_a_name, util_b_name) in combinations_with_replacement(self.utils_conf.keys(), 2):
                if util_a_name in self.sharing_factors_set \
                        or util_b_name in self.sharing_factors_set:
                    continue
                if util_a_name == util_b_name:
                    continue
                else:
                    model_key = f"({util_a_name}, {util_b_name})"
                    poten_ij, poten_ji = self.pp_models[model_key](utils[util_a_name], utils[util_b_name])
                    util_poten.setdefault(util_a_name, []).append(poten_ij)
                    util_poten.setdefault(util_b_name, []).append(poten_ji)

        # perform attention
        for util_name in self.utils_conf.keys():
            if self.prior_flag:
                prior = priors[util_name] \
                    if priors[util_name] is not None \
                    else torch.zeros_like(util_poten[util_name][0], requires_grad=False).cuda()

                util_poten[util_name].append(prior)

            util_poten[util_name] = torch.cat([p if len(p.size()) == 3 else p.unsqueeze(1)
                                               for p in util_poten[util_name]], dim=1)
            util_poten[util_name] = self.reduce_potentials[util_name](util_poten[util_name]).squeeze(1)
            util_poten[util_name] = F.softmax(util_poten[util_name], dim=1).unsqueeze(2)
            attention.append(torch.bmm(utils[util_name].transpose(1, 2), util_poten[util_name]).squeeze(2))

        return attention


class NaiveAttention(nn.Module):
    def __init__(self):
        """
            Used for ablation analysis - removing attention.
        """
        super(NaiveAttention, self).__init__()

    def forward(self, utils, priors):
        atten = []
        spatial_atten = []
        for u, p in zip(utils, priors):
            if type(u) is tuple:
                u = u[1]
                num_elements = u.shape[0]
                if p is not None:
                    u = u.view(-1, u.shape[-2], u.shape[-1])
                    p = p.view(-1, p.shape[-2], p.shape[-1])
                    spatial_atten.append(
                        torch.bmm(p.transpose(1, 2), u).squeeze(2).view(num_elements, -1, u.shape[-2], u.shape[-1]))
                else:
                    spatial_atten.append(u.mean(2))
                continue
            if p is not None:
                atten.append(torch.bmm(u.transpose(1, 2), p.unsqueeze(2)).squeeze(2))
            else:
                atten.append(u.mean(1))
        return atten, spatial_atten
