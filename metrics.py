import torch
import numpy as np
import math


def get_hit_k(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    return np.round(hit / pred_rank.shape[0], decimals=5)


def get_ndcg_k(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    return np.round(np.mean(ndcgs), decimals=5)


def evaluate(model, ratings, negatives, device, k_list, type_m="group"):
    hits_K, ndcgs_K = [], []

    topK_rank_array = np.zeros((len(ratings), max(k_list)))

    for idx in range(len(ratings)):
        user_test, item_test = [], []

        rating = ratings[idx]
        # candidate 0 is pos_item
        items = [rating[1]]
        items.extend(negatives[idx])

        item_test.append(items)
        user_test.append(np.full(len(items), rating[0]))

        users_var = torch.LongTensor(np.array(user_test)).view(-1).to(device)
        items_var = torch.LongTensor(np.array(item_test)).view(-1).to(device)

        predictions = model(users_var, items_var, type_m).squeeze()
        pred_score = predictions.data.cpu().numpy().reshape(1, -1)
        pred_rank = np.argsort(pred_score * -1, axis=1)

        topK_rank_array[idx, :] = pred_rank[0, : max(k_list)]

    for k in k_list:
        hits_K.append(get_hit_k(topK_rank_array, k))
        ndcgs_K.append(get_ndcg_k(topK_rank_array, k))

    return hits_K, ndcgs_K
