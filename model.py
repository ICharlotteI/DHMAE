import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InitModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)


class HGNN_Encoder(nn.Module):
    def __init__(
        self,
        user_hyper_graph,
        item_hyper_graph,
        full_hyper,
        emb_dim,
        num_layers,
        device,
    ):
        super(HGNN_Encoder, self).__init__()
        self.user_hyper, self.item_hyper, self.full_hyper_graph = (
            user_hyper_graph,
            item_hyper_graph,
            full_hyper,
        )
        self.hgnns = [HyperConvLayer(emb_dim).to(device) for _ in range(num_layers)]

    def forward(self, user_emb, item_emb, group_emb, num_users, num_items):
        init_ui_emb = torch.cat([user_emb, item_emb], dim=0)
        init_g_emb = group_emb
        final = [init_ui_emb]
        final_he = [init_g_emb]
        for hgnn in self.hgnns:
            ui_emb, g_emb = hgnn(
                user_emb,
                item_emb,
                init_g_emb,
                self.user_hyper,
                self.item_hyper,
                self.full_hyper_graph,
            )
            final.append(ui_emb)
            final_he.append(g_emb)

            user_emb, item_emb = torch.split(ui_emb, [num_users, num_items])

        final_emb = torch.sum(torch.stack(final), dim=0)
        final_he = torch.sum(torch.stack(final_he), dim=0)
        return torch.concat((final_emb, final_he), dim=0)


class MLP_Decoder(nn.Module):
    def __init__(self, emb_dim, num_layers, device, drop_ratio=0.0):
        super(MLP_Decoder, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layers = num_layers
        self.linears = [
            nn.Linear(emb_dim, emb_dim).to(device) for _ in range(num_layers)
        ]
        for linear in self.linears:
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)

    def forward(self, x):
        for index, linear in enumerate(self.linears):
            x = linear(x)
            if index < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, self.drop_ratio)
        return x


class AttentionLayer(InitModule):
    def __init__(self, emb_dim, drop_ratio=0.0):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.linear = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, 1),
        )
        self.init_weight()

    def forward(self, x, mask):
        bsz = x.shape[0]
        out = self.linear(x)
        out = out.view(bsz, -1)  # [bsz, max_len]
        out.masked_fill_(mask.bool(), -np.inf)
        weight = torch.softmax(out, dim=1)
        return weight


class Predictor(InitModule):
    def __init__(self, emb_dim, drop_ratio=0.0):
        super(Predictor, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, 1),
        )
        self.init_weight()

    def forward(self, x):
        return torch.sigmoid(self.MLP(x).squeeze())


class HyperConvLayer(InitModule):
    def __init__(self, emb_dim):
        super(HyperConvLayer, self).__init__()
        self.query_common = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1, bias=False),
        )
        self.user_lin = nn.Linear(2 * emb_dim, emb_dim)
        self.item_lin = nn.Linear(2 * emb_dim, emb_dim)

        self.init_weight()

    def forward(
        self,
        user_emb,
        item_emb,
        group_emb,
        user_hyper_graph,
        item_hyper_graph,
        full_hyper,
    ):
        user_msg = torch.sparse.mm(user_hyper_graph, user_emb)
        item_msg = torch.sparse.mm(item_hyper_graph, item_emb)

        att_common = torch.cat(
            [
                self.query_common(user_msg),
                self.query_common(item_msg),
            ],
            dim=-1,
        )
        weight_common = torch.softmax(att_common, dim=-1)
        common_msg = (
            weight_common[:, 0].unsqueeze(dim=1) * user_msg
            + weight_common[:, 1].unsqueeze(dim=1) * item_msg
        )
        user_msg = self.user_lin(
            torch.concat(((user_msg - common_msg), group_emb), dim=1)
        )
        item_msg = self.item_lin(
            torch.concat(((item_msg - common_msg), group_emb), dim=1)
        )
        msg = user_msg + item_msg + common_msg
        node_emb = torch.mm(full_hyper, msg)

        return node_emb, msg


class DHMAE(InitModule):
    def __init__(
        self,
        num_users,
        num_items,
        num_groups,
        group_member_dict,
        graph,
        mask_rate_mat,
        args,
    ):
        super(DHMAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups
        self.group_member_dict = group_member_dict
        self.mask_rate_mat = mask_rate_mat

        # Hyper-parameters
        self.emb_dim = args.emb_dim
        self.sce_alpha = args.sce_alpha
        self.drop_ratio = args.drop_ratio
        self.num_enc_layers = args.num_enc_layers
        self.num_dec_layers = args.num_dec_layers
        self.device = args.device

        # Embedding Layer
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        self.hgnn_encoder = HGNN_Encoder(
            graph["user_hg"],
            graph["item_hg"],
            graph["full_hg"],
            self.emb_dim,
            self.num_enc_layers,
            self.device,
        )
        self.mlp_decoder = MLP_Decoder(
            self.emb_dim, self.num_dec_layers, self.device, self.drop_ratio
        )

        self.regenerate = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.attention = AttentionLayer(self.emb_dim, self.drop_ratio)
        self.predictor = Predictor(self.emb_dim, self.drop_ratio)

        self.init_weight()

    def mask_encoding(self, x):
        mask_idxs = torch.bernoulli(1 - self.mask_rate_mat).nonzero()

        out_x = x.clone()
        out_x[mask_idxs] = 0.0 + 1e-8
        return (
            torch.split(out_x, [self.num_users, self.num_items, self.num_groups]),
            mask_idxs,
        )

    def ae_loss(self):
        init_x = torch.concat(
            (
                self.user_embedding.weight,
                self.item_embedding.weight,
                self.group_embedding.weight,
            ),
            dim=0,
        )
        # mask
        (user_emb, item_emb, group_emb), mask_idxs = self.mask_encoding(init_x)
        # encoder
        enc_x = self.hgnn_encoder(
            user_emb, item_emb, group_emb, self.num_users, self.num_items
        )
        # regenerate a new X to be masked
        masked_x = self.regenerate(enc_x)
        # second-time mask
        masked_x[mask_idxs] = 0.0 + 1e-8
        # decoder
        dec_x = self.mlp_decoder(masked_x)
        return (
            self.scaled_cosine_error(
                init_x[mask_idxs], dec_x[mask_idxs], self.sce_alpha
            ),
            enc_x,
        )

    def bpr_loss(
        self, user_inputs, pos_item_inputs, neg_item_inputs, enc_x=None, type="user"
    ):
        if type == "group":
            user_emb, item_emb, group_emb = torch.split(
                enc_x, [self.num_users, self.num_items, self.num_groups]
            )
            pos_g_emb = self.attentive_aggregate(
                user_emb, item_emb, group_emb, user_inputs, pos_item_inputs
            )
            neg_g_emb = self.attentive_aggregate(
                user_emb, item_emb, group_emb, user_inputs, neg_item_inputs
            )
            pos_i_emb = item_emb[pos_item_inputs]
            neg_i_emb = item_emb[neg_item_inputs]

            pos_scores = self.compute_score(pos_g_emb, pos_i_emb)
            neg_scores = self.compute_score(neg_g_emb, neg_i_emb)

        else:
            u_emb = self.user_embedding.weight[user_inputs]
            pos_i_emb = self.item_embedding.weight[pos_item_inputs]
            neg_i_emb = self.item_embedding.weight[neg_item_inputs]

            pos_scores = self.compute_score(u_emb, pos_i_emb)
            neg_scores = self.compute_score(u_emb, neg_i_emb)

        return torch.mean(F.softplus(neg_scores - pos_scores))

    def forward(self, user_inputs, item_inputs, type="user"):
        if type == "group":
            enc_x = self.hgnn_encoder(
                self.user_embedding.weight,
                self.item_embedding.weight,
                self.group_embedding.weight,
                self.num_users,
                self.num_items,
            )
            user_emb, item_emb, group_emb = torch.split(
                enc_x, [self.num_users, self.num_items, self.num_groups]
            )
            u_emb = self.attentive_aggregate(
                user_emb, item_emb, group_emb, user_inputs, item_inputs
            )
            i_emb = item_emb[item_inputs]
        else:
            u_emb = self.user_embedding.weight[user_inputs]
            i_emb = self.item_embedding.weight[item_inputs]

        return self.compute_score(u_emb, i_emb)

    def attentive_aggregate(
        self, user_emb, item_emb, group_emb, group_inputs, item_inputs
    ):
        i_emb = item_emb[item_inputs]
        g_emb = group_emb[group_inputs]

        member = []
        max_len = 0
        bsz = group_inputs.shape[0]
        member_masked = []
        for i in range(bsz):
            member.append(np.array(self.group_member_dict[group_inputs[i].item()]))
            max_len = max(max_len, len(self.group_member_dict[group_inputs[i].item()]))
        mask = np.zeros((bsz, max_len))
        for i, item in enumerate(member):
            cur_len = item.shape[0]
            member_masked.append(np.append(item, np.zeros(max_len - cur_len)))
            mask[i, cur_len:] = 1.0
        member_masked = torch.LongTensor(np.array(member_masked)).to(self.device)
        mask = torch.Tensor(mask).to(self.device)

        member_emb = user_emb[member_masked]
        item_emb_attn = i_emb.unsqueeze(1).expand(bsz, max_len, -1)
        at_wt = self.attention(torch.cat((member_emb, item_emb_attn), dim=2), mask)
        g_emb_with_attention = torch.matmul(at_wt.unsqueeze(1), member_emb).squeeze()
        return g_emb_with_attention + g_emb

    def scaled_cosine_error(self, x, y, alpha=1):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def compute_score(self, u_emb, i_emb):
        element_emb = torch.mul(u_emb, i_emb)
        return self.predictor(torch.cat((element_emb, u_emb, i_emb), dim=1))
