"""LightGCN — 轻量级图卷积推荐模型 (PyTorch Geometric)"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.sparse as sp

from .base import BaseRecommender
import logging

logger = logging.getLogger(__name__)


def build_adj_matrix(n_users, n_items, ratings_df, user_col="user_idx", item_col="movie_idx"):
    """构建用户-物品二部图的邻接矩阵 (归一化)"""
    n_total = n_users + n_items
    rows = ratings_df[user_col].values
    cols = ratings_df[item_col].values + n_users  # item节点偏移

    # 无向图：user→item 和 item→user
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones(len(all_rows), dtype=np.float32)

    adj = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n_total, n_total))

    # 对称归一化: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt

    return norm_adj


def sparse_mx_to_torch(sparse_mx):
    """将scipy稀疏矩阵转为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.vstack([sparse_mx.row, sparse_mx.col]))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class LightGCNModel(BaseRecommender):
    """LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"""

    def __init__(self):
        super().__init__(name="lightgcn")
        cfg = self.config.get("models.lightgcn", {})
        self.embedding_dim = cfg.get("embedding_dim", 64)
        self.n_layers = cfg.get("n_layers", 3)
        self.reg_weight = cfg.get("reg_weight", 1e-4)
        self.lr = cfg.get("lr", 0.001)
        self.epochs = cfg.get("epochs", 100)
        self.batch_size = cfg.get("batch_size", 2048)
        self.patience = cfg.get("early_stop_patience", 10)

        self.user_emb = None
        self.item_emb = None
        self.norm_adj = None
        self._emb_state_bytes: bytes = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.user_emb is not None and self.item_emb is not None:
            import io
            buffer = io.BytesIO()
            torch.save({
                "user_emb": self.user_emb.state_dict(),
                "item_emb": self.item_emb.state_dict(),
            }, buffer)
            state["_emb_state_bytes"] = buffer.getvalue()
        state["user_emb"] = None
        state["item_emb"] = None
        state["norm_adj"] = None
        state["final_user_emb"] = getattr(self, "final_user_emb", None)
        state["final_item_emb"] = getattr(self, "final_item_emb", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        state_bytes = state.get("_emb_state_bytes")
        if state_bytes is not None:
            import io
            self.user_emb = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
            self.item_emb = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
            buffer = io.BytesIO(state_bytes)
            ckpt = torch.load(buffer, map_location=self.device)
            self.user_emb.load_state_dict(ckpt["user_emb"])
            self.item_emb.load_state_dict(ckpt["item_emb"])
        self.norm_adj = None

    def train(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None):
        self.load_data(train_df, val_df)
        n_total = self.n_users + self.n_items

        # 构建归一化邻接矩阵
        logger.info("[LightGCN] 构建二部图邻接矩阵...")
        adj = build_adj_matrix(self.n_users, self.n_items, self.train_data)
        self.norm_adj = sparse_mx_to_torch(adj).to(self.device)

        # 初始化嵌入
        self.user_emb = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

        params = list(self.user_emb.parameters()) + list(self.item_emb.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # 训练用的交互对
        train_users = torch.LongTensor(self.train_data["user_idx"].values).to(self.device)
        train_items = torch.LongTensor(self.train_data["movie_idx"].values).to(self.device)
        n_train = len(train_users)

        best_loss = float("inf")
        patience_counter = 0

        logger.info(f"[LightGCN] 开始训练 ({self.epochs} epochs, {self.n_layers} layers)")

        for epoch in range(self.epochs):
            self.user_emb.train()
            self.item_emb.train()

            # 全量嵌入传播
            all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            embs = [all_emb]

            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.norm_adj, all_emb)
                embs.append(all_emb)

            # 层聚合（均值）
            final_emb = torch.stack(embs, dim=0).mean(dim=0)
            u_emb, i_emb = torch.split(final_emb, [self.n_users, self.n_items])

            # BPR 损失
            perm = torch.randperm(n_train)[:self.batch_size * 10]  # 采样
            batch_u = train_users[perm]
            batch_i = train_items[perm]

            # 负采样
            batch_j = torch.randint(0, self.n_items, (len(batch_u),)).to(self.device)

            u_emb_batch = u_emb[batch_u]
            pos_emb = i_emb[batch_i]
            neg_emb = i_emb[batch_j]

            pos_scores = (u_emb_batch * pos_emb).sum(dim=1)
            neg_scores = (u_emb_batch * neg_emb).sum(dim=1)

            bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            # 正则化
            reg_loss = (
                self.reg_weight
                * (u_emb_batch.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))
                / len(batch_u)
            )
            loss = bpr_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"[LightGCN] Epoch {epoch+1}/{self.epochs}, BPR Loss: {loss.item():.4f}")

            if loss.item() < best_loss - 1e-4:
                best_loss = loss.item()
                patience_counter = 0
                torch.save(
                    {"user_emb": self.user_emb.state_dict(), "item_emb": self.item_emb.state_dict()},
                    self.model_dir / "lightgcn_best.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"[LightGCN] Early stop at epoch {epoch+1}")
                    break

        # 加载最佳
        best_path = self.model_dir / "lightgcn_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device)
            self.user_emb.load_state_dict(ckpt["user_emb"])
            self.item_emb.load_state_dict(ckpt["item_emb"])

        # 计算最终嵌入
        with torch.no_grad():
            all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
            embs = [all_emb]
            for _ in range(self.n_layers):
                all_emb = torch.sparse.mm(self.norm_adj, all_emb)
                embs.append(all_emb)
            final_emb = torch.stack(embs, dim=0).mean(dim=0)
            self.final_user_emb, self.final_item_emb = torch.split(
                final_emb, [self.n_users, self.n_items]
            )

        logger.info("[LightGCN] 训练完成")

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """用嵌入内积预测评分（归一化到1-5）"""
        self.user_emb.eval()
        self.item_emb.eval()
        with torch.no_grad():
            u_vec = self.final_user_emb[torch.LongTensor(user_ids).to(self.device)]
            i_vec = self.final_item_emb[torch.LongTensor(item_ids).to(self.device)]
            # 内积越大 → 评分越高，用sigmoid映射到[0,1]再转[1,5]
            scores = torch.sigmoid((u_vec * i_vec).sum(dim=1)).cpu().numpy()
        return 1 + 4 * scores

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        self.user_emb.eval()
        self.item_emb.eval()

        seen = set()
        if exclude_seen and self.train_data is not None:
            user_data = self.train_data[self.train_data["user_idx"] == user_idx]
            seen = set(user_data["movie_idx"].values)

        candidates = np.array([i for i in range(self.n_items) if i not in seen])
        if len(candidates) == 0:
            return []

        user_arr = np.full(len(candidates), user_idx, dtype=np.int64)
        scores = self.predict(user_arr, candidates)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(candidates[i]), float(scores[i])) for i in top_indices]
