from torch.utils.data import TensorDataset, DataLoader
from datautil import *


class GroupDataset(object):
    def __init__(self, dataset):
        print(f"[{dataset}] loading...".upper())
        user_path, group_path = (
            f"./data/{dataset}/userRating",
            f"./data/{dataset}/groupRating",
        )
        # User Data
        if dataset == "MafengwoS":
            # num_users 11027 num_items 1236 num_groups 1215
            self.user_train_matrix = load_rating_file_to_matrix(
                user_path + "Train.txt", num_users=11026, num_items=1235
            )
        else:
            self.user_train_matrix = load_rating_file_to_matrix(user_path + "Train.txt")
        self.num_users, self.num_items = self.user_train_matrix.shape
        self.user_test_ratings = load_test_ratings(user_path + "Test.txt")
        self.user_test_negatives = load_test_negatives(
            self.user_train_matrix, self.user_test_ratings
        )
        # Group Data
        self.group_member_dict = load_group_member_to_dict(
            f"./data/{dataset}/groupMember.txt"
        )
        self.num_groups = len(self.group_member_dict)
        self.group_train_matrix = load_rating_file_to_matrix(
            group_path + "Train.txt",
            num_users=self.num_groups - 1,
            num_items=self.num_items - 1,
        )

        self.group_test_ratings = load_test_ratings(group_path + "Test.txt")
        self.group_test_negatives = load_test_negatives(
            self.group_train_matrix, self.group_test_ratings
        )
        print(
            f" #Users {self.num_users}, #Items {self.num_items}, #Groups {self.num_groups}"
        )

        # train dataset info
        print(
            f"UserItemTrain: {self.user_train_matrix.count_nonzero()} interactions, "
            f"sparsity ratio: {(1 - (self.user_train_matrix.count_nonzero() / (self.num_users * self.num_items))):.5f}"
        )
        print(
            f"GroupItemTrain: {self.group_train_matrix.count_nonzero()} interactions, "
            f"sparsity ratio: {(1 - (self.group_train_matrix.count_nonzero() / (self.num_groups * self.num_items))):.5f}"
        )

        # build graph
        self.mask_rate_mat = get_uig_mask_rate(
            f"./data/{dataset}/uig_entity_value.npz",
            self.group_member_dict,
            self.group_train_matrix,
            self.user_train_matrix,
            self.num_users,
            self.num_items,
        )

        (
            self.user_hyper_graph,
            self.item_hyper_graph,
            self.full_hg,
        ) = build_hyper_graph(
            self.group_member_dict,
            self.group_train_matrix,
            self.num_users,
            self.num_items,
            self.num_groups,
        )

        print(f"[{dataset}] loaded!".upper())

    def get_train_instances(self, train, num_negatives):
        users, pos_items, neg_items = [], [], []

        num_users, num_items = train.shape[0], train.shape[1]
        for u, i in train.keys():
            for _ in range(num_negatives):
                users.append(u)
                pos_items.append(i)

                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                neg_items.append(j)
        pos_neg_items = [
            [pos_item, neg_item] for pos_item, neg_item in zip(pos_items, neg_items)
        ]
        return users, pos_neg_items

    def get_user_train_dataloader(self, batch_size, num_negatives):
        users, pos_neg_items = self.get_train_instances(
            self.user_train_matrix, num_negatives
        )
        train_data = TensorDataset(
            torch.LongTensor(users), torch.LongTensor(pos_neg_items)
        )
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)

    def get_group_train_dataloader(self, batch_size, num_negatives):
        groups, pos_neg_items = self.get_train_instances(
            self.group_train_matrix, num_negatives
        )
        train_data = TensorDataset(
            torch.LongTensor(groups), torch.LongTensor(pos_neg_items)
        )
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)
