import os
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import torch


def load_rating_file_to_matrix(filename, num_users=None, num_items=None):
    if num_users is None:
        num_users, num_items = 0, 0

    lines = open(filename, "r").readlines()
    for line in lines:
        contents = line.split()
        u, i = int(contents[0]), int(contents[1])
        num_users = max(num_users, u)
        num_items = max(num_items, i)

    train_mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)

    for line in lines:
        contents = line.split()
        if len(contents) > 2:
            u, i, rating = int(contents[0]), int(contents[1]), int(contents[2])
            if rating > 0:
                train_mat[u, i] = 1.0
        else:
            u, i = int(contents[0]), int(contents[1])
            train_mat[u, i] = 1.0
    return train_mat


def load_test_negatives(train, ratings):
    negative_list = []

    for i in range(len(ratings)):
        # all uninteracted items for current user
        non_inter_items = np.argwhere(train[ratings[i][0]].toarray()[0] == 0).squeeze()
        # remove pos_item (since the evaluation puts it in candidate 0, based on whether there is candidate 0 in topK)
        negative_list.append(non_inter_items[non_inter_items != ratings[i][1]].tolist())

    return negative_list


def load_test_ratings(filename):
    rating_list = []
    lines = open(filename, "r").readlines()

    for line in lines:
        contents = line.split()
        rating_list.append([int(contents[0]), int(contents[1])])
    return rating_list


def load_group_member_to_dict(user_in_group_path):
    group_member_dict = defaultdict(list)
    lines = open(user_in_group_path, "r").readlines()

    for line in lines:
        contents = line.split()
        group = int(contents[0])
        for member in contents[1].split(","):
            group_member_dict[group].append(int(member))
    return group_member_dict


def get_uig_mask_rate(
    filename, group_member_dict, group_train, user_train, num_users, num_items
):
    if os.path.exists(filename):
        loaded_data = np.load(filename)
        u_array, i_array, g_array = (
            loaded_data["u_array"],
            loaded_data["i_array"],
            loaded_data["g_array"],
        )
    else:
        u_list = [0] * num_users
        for user_id in range(num_users):
            u_list[user_id] += user_train[user_id].count_nonzero()
        for group_id, user_ids in group_member_dict.items():
            for user_id in user_ids:
                u_list[user_id] += 1

        i_list = [0] * num_items
        for item_id in range(num_items):
            i_list[item_id] += (
                user_train[:, item_id].count_nonzero()
                + group_train[:, item_id].count_nonzero()
            )

        g_list = []
        for group_id, user_ids in group_member_dict.items():
            g_list.append(len(user_ids) + group_train[group_id].count_nonzero())

        u_array, i_array, g_array = np.array(u_list), np.array(i_list), np.array(g_list)
        np.savez_compressed(filename, u_array=u_array, i_array=i_array, g_array=g_array)

    mask_rate_mat = 1.0 / torch.tensor(
        np.concatenate((u_array, i_array, g_array), axis=0)
    )
    mask_rate_mat[mask_rate_mat == float("inf")] = 0
    return mask_rate_mat


def build_hyper_graph(
    group_member_dict,
    group_train,
    num_users,
    num_items,
    num_groups,
    group_item_dict=None,
):
    if group_item_dict is None:
        group_item_dict = defaultdict(list)

        row, col = group_train.nonzero()
        row = np.squeeze(row)
        col = np.squeeze(col)
        for group, item in zip(row, col):
            group_item_dict[group].append(item)

    def _prepare(group_dict, rows, axis=0):
        nodes, groups = [], []

        for group_id in range(num_groups):
            groups.extend([group_id] * len(group_dict[group_id]))
            nodes.extend(group_dict[group_id])

        hyper_graph = sp.csr_matrix(
            (np.ones(len(nodes)), (nodes, groups)), shape=(rows, num_groups)
        )
        hyper_deg = np.array(hyper_graph.sum(axis=axis)).squeeze()
        hyper_deg[hyper_deg == 0.0] = 1
        hyper_deg = sp.diags(1.0 / hyper_deg)
        return hyper_graph, hyper_deg

    user_hg, user_hg_deg = _prepare(group_member_dict, num_users)
    item_hg, item_hg_deg = _prepare(group_item_dict, num_items)

    for group_id, items in group_item_dict.items():
        group_item_dict[group_id] = [item + num_users for item in items]
    group_data = [
        group_member_dict[group_id] + group_item_dict[group_id]
        for group_id in range(num_groups)
    ]
    full_hg, hg_dg = _prepare(group_data, num_users + num_items, axis=1)

    user_hyper_graph = torch.sparse.mm(
        convert_sp_mat_to_sp_tensor(user_hg_deg),
        convert_sp_mat_to_sp_tensor(user_hg).t(),
    )
    item_hyper_graph = torch.sparse.mm(
        convert_sp_mat_to_sp_tensor(item_hg_deg),
        convert_sp_mat_to_sp_tensor(item_hg).t(),
    )
    full_hyper_graph = torch.sparse.mm(
        convert_sp_mat_to_sp_tensor(hg_dg), convert_sp_mat_to_sp_tensor(full_hg)
    )

    return user_hyper_graph, item_hyper_graph, full_hyper_graph


def convert_sp_mat_to_sp_tensor(x):
    coo = x.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
