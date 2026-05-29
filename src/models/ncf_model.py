"""Neural Collaborative Filtering — PyTorch 实现"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .base import BaseRecommender
import logging

logger = logging.getLogger(__name__)


class NCFDataset(Dataset):
    """NCF训练数据集 — 正负样本"""

    def __init__(self, ratings_df, n_items, neg_ratio=4):
        self.users = ratings_df["user_idx"].values.astype(np.int64)
        self.items = ratings_df["movie_idx"].values.astype(np.int64)
        self.labels = np.ones(len(ratings_df), dtype=np.float32)
        self.n_items = n_items
        self.neg_ratio = neg_ratio

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]

        # 随机负采样
        neg_items = np.random.randint(0, self.n_items, size=self.neg_ratio)
        # 避免采样到正样本（简单处理）
        while pos_item in neg_items:
            neg_items[neg_items == pos_item] = np.random.randint(0, self.n_items)

        return (
            torch.LongTensor([user] * (1 + self.neg_ratio)),
            torch.LongTensor([pos_item] + list(neg_items)),
            torch.FloatTensor([1.0] + [0.0] * self.neg_ratio),
        )


class NCF(nn.Module):
    """Neural Collaborative Filtering: GMF + MLP 双塔融合"""

    def __init__(self, n_users, n_items, embedding_dim=64,
                 mlp_layers=(128, 64, 32, 16), dropout=0.2):
        super().__init__()

        # GMF 嵌入
        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP 嵌入
        self.user_emb_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(n_items, embedding_dim)

        # MLP 层
        mlp_modules = []
        input_dim = embedding_dim * 2
        for out_dim in mlp_layers:
            mlp_modules.extend([
                nn.Linear(input_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # 融合层
        fusion_dim = embedding_dim + mlp_layers[-1]
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF path
        ug = self.user_emb_gmf(user_ids)
        ig = self.item_emb_gmf(item_ids)
        gmf_vec = ug * ig  # element-wise product

        # MLP path
        um = self.user_emb_mlp(user_ids)
        im = self.item_emb_mlp(item_ids)
        mlp_input = torch.cat([um, im], dim=-1)
        mlp_vec = self.mlp(mlp_input)

        # Fusion
        fusion = torch.cat([gmf_vec, mlp_vec], dim=-1)
        return self.predictor(fusion).squeeze()


class NCFModel(BaseRecommender):
    """NCF推荐模型"""

    def __init__(self):
        super().__init__(name="ncf")
        cfg = self.config.get("models.ncf", {})
        self.embedding_dim = cfg.get("embedding_dim", 64)
        self.mlp_layers = cfg.get("mlp_layers", [128, 64, 32, 16])
        self.dropout = cfg.get("dropout", 0.2)
        self.neg_ratio = cfg.get("neg_ratio", 4)
        self.batch_size = cfg.get("batch_size", 1024)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 0.001)
        self.patience = cfg.get("early_stop_patience", 5)

        self.model: NCF = None
        self.optimizer = None
        self._state_dict_bytes: bytes = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            import io
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            state["_state_dict_bytes"] = buffer.getvalue()
        state["model"] = None
        state["optimizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        state_bytes = state.get("_state_dict_bytes")
        if state_bytes is not None:
            import io
            self.model = NCF(
                n_users=self.n_users,
                n_items=self.n_items,
                embedding_dim=self.embedding_dim,
                mlp_layers=self.mlp_layers,
                dropout=self.dropout,
            ).to(self.device)
            buffer = io.BytesIO(state_bytes)
            self.model.load_state_dict(torch.load(buffer, map_location=self.device))

    def train(self, train_df: pd.DataFrame = None, val_df: pd.DataFrame = None):
        self.load_data(train_df, val_df)

        self.model = NCF(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_dim=self.embedding_dim,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = NCFDataset(self.train_data, self.n_items, self.neg_ratio)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        patience_counter = 0

        logger.info(f"[NCF] 开始训练 ({self.epochs} epochs, batch={self.batch_size})")

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for users, items, labels in dataloader:
                users = users.view(-1).to(self.device)
                items = items.view(-1).to(self.device)
                labels = labels.view(-1).to(self.device)

                preds = self.model(users, items)
                loss = criterion(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            if (epoch + 1) % 5 == 0:
                logger.info(f"[NCF] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_dir / "ncf_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"[NCF] Early stop at epoch {epoch+1}")
                    break

        # 加载最佳模型
        best_path = self.model_dir / "ncf_best.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        logger.info("[NCF] 训练完成")

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(user_ids), self.batch_size):
                batch_u = torch.LongTensor(user_ids[i:i+self.batch_size]).to(self.device)
                batch_i = torch.LongTensor(item_ids[i:i+self.batch_size]).to(self.device)
                p = self.model(batch_u, batch_i).cpu().numpy()
                preds.append(p)
        raw = np.concatenate(preds)
        # 从[0,1]映射到[1,5]
        return 1 + 4 * raw

    def recommend(self, user_idx: int, top_k: int = 10,
                  exclude_seen: bool = True) -> list:
        self.model.eval()
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
