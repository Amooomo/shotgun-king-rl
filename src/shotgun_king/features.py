import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)


class ShotgunCNNExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        channels, height, width = (
            board_space.shape
        )

        # =============================
        # Spatial board encoder
        # =============================

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Flatten(),
        )

        # 自动计算 CNN 输出维度
        with torch.no_grad():

            sample = torch.zeros(
                1,
                channels,
                height,
                width,
            )

            cnn_dim = (
                self.cnn(sample)
                .shape[1]
            )

        stats_space = (
            observation_space.spaces[
                "stats"
            ]
        )

        stats_dim = int(
            np.prod(
                stats_space.shape
            )
        )

        # =============================
        # Board + stats fusion
        # =============================

        self.fusion = nn.Sequential(

            nn.Linear(
                cnn_dim + stats_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        board_features = (
            self.cnn(board)
        )

        combined = torch.cat(
            [
                board_features,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            combined
        )

class ShotgunSmallCNNExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        features_dim=128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        channels, height, width = (
            board_space.shape
        )

        self.cnn = nn.Sequential(

            nn.Conv2d(
                channels,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                channels,
                height,
                width,
            )

            cnn_dim = (
                self.cnn(dummy)
                .shape[1]
            )

        stats_dim = int(
            np.prod(
                observation_space
                .spaces["stats"]
                .shape
            )
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                cnn_dim + stats_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        board_features = (
            self.cnn(board)
        )

        features = torch.cat(
            [
                board_features,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            features
        )
    
class ShotgunUNetLiteExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        features_dim=128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        in_channels, height, width = (
            board_space.shape
        )

        # =====================================
        # Encoder Level 1
        # 8 × 8
        # =====================================

        self.enc1 = nn.Sequential(

            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        # =====================================
        # Encoder Level 2
        # 8×8 → 4×4
        # =====================================

        self.down1 = nn.Sequential(

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        # =====================================
        # Global Bottleneck
        # 4×4 → 2×2
        # =====================================

        self.down2 = nn.Sequential(

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        # =====================================
        # Decoder Level 2
        # Global → Medium
        # =====================================

        self.dec2 = nn.Sequential(

            nn.Conv2d(
                64 + 32,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        # =====================================
        # Decoder Level 1
        # Medium → Local
        # =====================================

        self.dec1 = nn.Sequential(

            nn.Conv2d(
                32 + 16,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        # =====================================
        # Multi-scale pooling
        # =====================================

        self.local_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.medium_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        stats_dim = int(
            np.prod(
                observation_space
                .spaces["stats"]
                .shape
            )
        )

        # Global 2×2 不做 GAP。
        #
        # 保留四个象限的位置关系。
        global_dim = (
            64 * 2 * 2
        )

        local_dim = 16
        medium_dim = 32

        fusion_dim = (
            local_dim
            + medium_dim
            + global_dim
            + stats_dim
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                fusion_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        # ================================
        # Encoder
        # ================================

        x1 = self.enc1(
            board
        )
        # B × 16 × 8 × 8

        x2 = self.down1(
            x1
        )
        # B × 32 × 4 × 4

        x3 = self.down2(
            x2
        )
        # B × 64 × 2 × 2

        # ================================
        # Decoder + skip
        # ================================

        up2 = nn.functional.interpolate(
            x3,
            size=x2.shape[-2:],
            mode="nearest",
        )

        d2 = torch.cat(
            [
                up2,
                x2,
            ],
            dim=1,
        )

        d2 = self.dec2(
            d2
        )
        # B × 32 × 4 × 4

        up1 = nn.functional.interpolate(
            d2,
            size=x1.shape[-2:],
            mode="nearest",
        )

        d1 = torch.cat(
            [
                up1,
                x1,
            ],
            dim=1,
        )

        d1 = self.dec1(
            d1
        )
        # B × 16 × 8 × 8

        # ================================
        # Multi-scale feature aggregation
        # ================================

        local_feature = (
            self.local_pool(d1)
            .flatten(1)
        )
        # 16

        medium_feature = (
            self.medium_pool(d2)
            .flatten(1)
        )
        # 32

        global_feature = (
            x3.flatten(1)
        )
        # 64 × 2 × 2 = 256

        combined = torch.cat(
            [
                local_feature,
                medium_feature,
                global_feature,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            combined
        )

class ShotgunCoordCNNExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        features_dim=128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        channels, height, width = (
            board_space.shape
        )

        # ============================
        # Coordinate channels
        # ============================

        row_coord = (
            torch.linspace(
                -1.0,
                1.0,
                steps=height,
            )
            .view(
                1, 1, height, 1
            )
            .expand(
                1, 1, height, width
            )
            .clone()
        )

        col_coord = (
            torch.linspace(
                -1.0,
                1.0,
                steps=width,
            )
            .view(
                1, 1, 1, width
            )
            .expand(
                1, 1, height, width
            )
            .clone()
        )

        self.register_buffer(
            "row_coord",
            row_coord,
        )

        self.register_buffer(
            "col_coord",
            col_coord,
        )

        # 4 piece channels
        # +
        # row channel
        # +
        # col channel
        #
        # = 6 channels

        coord_channels = (
            channels + 2
        )

        # ============================
        # Small CNN
        # ============================

        self.cnn = nn.Sequential(

            nn.Conv2d(
                coord_channels,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Flatten(),
        )

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                coord_channels,
                height,
                width,
            )

            cnn_dim = (
                self.cnn(dummy)
                .shape[1]
            )

        stats_dim = int(
            np.prod(
                observation_space
                .spaces["stats"]
                .shape
            )
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                cnn_dim + stats_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        batch_size = board.shape[0]

        row = self.row_coord.expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        col = self.col_coord.expand(
            batch_size,
            -1,
            -1,
            -1,
        )

        board_with_coords = (
            torch.cat(
                [
                    board,
                    row,
                    col,
                ],
                dim=1,
            )
        )

        board_features = self.cnn(
            board_with_coords
        )

        combined = torch.cat(
            [
                board_features,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            combined
        )

class ShotgunTransformerExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        features_dim=128,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        channels, height, width = (
            board_space.shape
        )

        assert channels == 4

        self.height = height
        self.width = width
        self.n_cells = (
            height * width
        )

        # =================================
        # Piece embedding
        #
        # 0 = EMPTY
        # 1 = BLACK_KING
        # 2 = WHITE_KING
        # 3 = WHITE_ROOK
        # 4 = WHITE_PAWN
        # =================================

        self.piece_embedding = (
            nn.Embedding(
                5,
                d_model,
            )
        )

        # =================================
        # Absolute spatial position
        # =================================

        self.row_embedding = (
            nn.Embedding(
                height,
                d_model,
            )
        )

        self.col_embedding = (
            nn.Embedding(
                width,
                d_model,
            )
        )

        # =================================
        # CLS token
        # =================================

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02,
        )

        # =================================
        # Fixed row / column indices
        # =================================

        row_ids = (
            torch.arange(height)
            .repeat_interleave(width)
        )

        col_ids = (
            torch.arange(width)
            .repeat(height)
        )

        self.register_buffer(
            "row_ids",
            row_ids,
            persistent=False,
        )

        self.register_buffer(
            "col_ids",
            col_ids,
            persistent=False,
        )

        # =================================
        # Transformer
        #
        # 使用 ModuleList 独立创建，
        # 避免所有 EncoderLayer 具有完全相同
        # 的初始参数。
        # =================================

        self.layers = nn.ModuleList(

            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,

                    dim_feedforward=(
                        dim_feedforward
                    ),

                    dropout=0.0,

                    activation="gelu",

                    batch_first=True,

                    norm_first=True,
                )

                for _ in range(
                    num_layers
                )
            ]
        )

        self.final_norm = (
            nn.LayerNorm(
                d_model
            )
        )

        stats_dim = int(
            np.prod(
                observation_space
                .spaces["stats"]
                .shape
            )
        )

        # =================================
        # Board relation + stats
        # =================================

        self.fusion = nn.Sequential(

            nn.Linear(
                d_model + stats_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def _board_to_piece_ids(
        self,
        board,
    ):

        # board:
        # B × 4 × 8 × 8

        occupied = (
            board.sum(
                dim=1
            )
            > 0
        )

        # 0~3 channel index
        channel_id = (
            board.argmax(
                dim=1
            )
        )

        # 转为 Piece ID 1~4
        piece_id = (
            channel_id + 1
        )

        # Empty = 0
        piece_id = torch.where(
            occupied,
            piece_id,
            torch.zeros_like(
                piece_id
            ),
        )

        return piece_id.long()

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        batch_size = (
            board.shape[0]
        )

        # =================================
        # Board → piece IDs
        # =================================

        piece_ids = (
            self._board_to_piece_ids(
                board
            )
        )

        piece_ids = (
            piece_ids.reshape(
                batch_size,
                self.n_cells,
            )
        )

        # =================================
        # Token embedding
        # =================================

        piece_tokens = (
            self.piece_embedding(
                piece_ids
            )
        )

        row_pos = (
            self.row_embedding(
                self.row_ids
            )
            .unsqueeze(0)
        )

        col_pos = (
            self.col_embedding(
                self.col_ids
            )
            .unsqueeze(0)
        )

        tokens = (
            piece_tokens
            + row_pos
            + col_pos
        )

        # =================================
        # CLS
        # =================================

        cls = (
            self.cls_token.expand(
                batch_size,
                -1,
                -1,
            )
        )

        tokens = torch.cat(
            [
                cls,
                tokens,
            ],
            dim=1,
        )

        # B × 65 × d_model

        # =================================
        # Self Attention
        # =================================

        for layer in self.layers:

            tokens = layer(
                tokens
            )

        tokens = self.final_norm(
            tokens
        )

        # =================================
        # CLS = whole-board representation
        # =================================

        board_feature = (
            tokens[:, 0]
        )

        combined = torch.cat(
            [
                board_feature,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            combined
        )

class ShotgunPieceTransformerExtractor(
    BaseFeaturesExtractor,
):

    def __init__(
        self,
        observation_space,
        features_dim=128,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    ):

        super().__init__(
            observation_space,
            features_dim,
        )

        board_space = (
            observation_space.spaces[
                "board"
            ]
        )

        channels, height, width = (
            board_space.shape
        )

        assert channels == 4

        self.height = height
        self.width = width

        # ---------------------------------
        # Piece IDs
        #
        # 0 = PAD
        # 1 = BLACK_KING
        # 2 = WHITE_KING
        # 3 = WHITE_ROOK
        # 4 = WHITE_PAWN
        # ---------------------------------

        self.piece_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=d_model,
            padding_idx=0,
        )

        # 多留一个 index 给 PAD
        self.row_embedding = nn.Embedding(
            num_embeddings=height + 1,
            embedding_dim=d_model,
            padding_idx=height,
        )

        self.col_embedding = nn.Embedding(
            num_embeddings=width + 1,
            embedding_dim=d_model,
            padding_idx=width,
        )

        self.pad_row = height
        self.pad_col = width

        # ---------------------------------
        # CLS
        # ---------------------------------

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        nn.init.normal_(
            self.cls_token,
            mean=0.0,
            std=0.02,
        )

        # ---------------------------------
        # Transformer
        # ---------------------------------

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=(
                        dim_feedforward
                    ),
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )

                for _ in range(
                    num_layers
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        stats_dim = int(
            np.prod(
                observation_space
                .spaces["stats"]
                .shape
            )
        )

        self.fusion = nn.Sequential(

            nn.Linear(
                d_model + stats_dim,
                features_dim,
            ),

            nn.ReLU(),
        )

    def _extract_piece_tokens(
        self,
        board,
    ):

        # board:
        # B × 4 × 8 × 8

        batch_size = board.shape[0]

        flat = board.reshape(
            batch_size,
            4,
            -1,
        )

        # =================================
        # Black King
        # =================================

        bk_index = (
            flat[:, 0]
            .argmax(dim=1)
            .unsqueeze(1)
        )

        bk_valid = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=board.device,
        )

        # =================================
        # White King
        # =================================

        wk_index = (
            flat[:, 1]
            .argmax(dim=1)
            .unsqueeze(1)
        )

        wk_valid = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=board.device,
        )

        # =================================
        # Rooks: max 2
        # =================================

        rook_values, rook_index = (
            torch.topk(
                flat[:, 2],
                k=2,
                dim=1,
            )
        )

        rook_valid = (
            rook_values > 0
        )

        # =================================
        # Pawns: max 4
        # =================================

        pawn_values, pawn_index = (
            torch.topk(
                flat[:, 3],
                k=4,
                dim=1,
            )
        )

        pawn_valid = (
            pawn_values > 0
        )

        # =================================
        # Combine positions
        # =================================

        positions = torch.cat(
            [
                bk_index,
                wk_index,
                rook_index,
                pawn_index,
            ],
            dim=1,
        )

        valid = torch.cat(
            [
                bk_valid,
                wk_valid,
                rook_valid,
                pawn_valid,
            ],
            dim=1,
        )

        # 8 token slots

        piece_ids = torch.tensor(
            [
                1,  # BK
                2,  # WK
                3,  # R
                3,  # R
                4,  # P
                4,  # P
                4,  # P
                4,  # P
            ],
            device=board.device,
            dtype=torch.long,
        )

        piece_ids = (
            piece_ids
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
            .clone()
        )

        # 不存在的 slot → PAD
        piece_ids = torch.where(
            valid,
            piece_ids,
            torch.zeros_like(
                piece_ids
            ),
        )

        rows = (
            positions
            // self.width
        )

        cols = (
            positions
            % self.width
        )

        rows = torch.where(
            valid,
            rows,
            torch.full_like(
                rows,
                self.pad_row,
            ),
        )

        cols = torch.where(
            valid,
            cols,
            torch.full_like(
                cols,
                self.pad_col,
            ),
        )

        return (
            piece_ids,
            rows,
            cols,
            valid,
        )

    def forward(
        self,
        observations,
    ):

        board = (
            observations["board"]
            .float()
        )

        stats = (
            observations["stats"]
            .float()
        )

        batch_size = (
            board.shape[0]
        )

        (
            piece_ids,
            rows,
            cols,
            valid,
        ) = self._extract_piece_tokens(
            board
        )

        # =================================
        # Piece representation
        # =================================

        tokens = (
            self.piece_embedding(
                piece_ids
            )
            +
            self.row_embedding(
                rows
            )
            +
            self.col_embedding(
                cols
            )
        )

        # B × 8 × d_model

        # =================================
        # CLS
        # =================================

        cls = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            [
                cls,
                tokens,
            ],
            dim=1,
        )

        # =================================
        # Padding mask
        #
        # True = ignore
        # =================================

        cls_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=board.device,
        )

        padding_mask = torch.cat(
            [
                cls_mask,
                ~valid,
            ],
            dim=1,
        )

        # =================================
        # Transformer
        # =================================

        for layer in self.layers:

            tokens = layer(
                tokens,
                src_key_padding_mask=(
                    padding_mask
                ),
            )

        tokens = self.final_norm(
            tokens
        )

        board_feature = (
            tokens[:, 0]
        )

        combined = torch.cat(
            [
                board_feature,
                stats,
            ],
            dim=1,
        )

        return self.fusion(
            combined
        )