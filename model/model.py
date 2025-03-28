import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Optional


class ReversiModel:
    """リバーシAIのモデルクラス"""

    def __init__(self, model_path: Optional[str] = None):
        """モデルを初期化または読み込む"""
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """モデルを構築する"""
        # 入力層: 8x8の盤面を3チャンネル（プレイヤー1の石、プレイヤー2の石、空きマス）で表現
        input_layer = tf.keras.layers.Input(shape=(8, 8, 3))

        # 畳み込み層
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # 特徴抽出
        features = tf.keras.layers.Flatten()(x)
        features = tf.keras.layers.Dense(256, activation='relu')(features)
        features = tf.keras.layers.BatchNormalization()(features)
        features = tf.keras.layers.Dropout(0.3)(features)

        # 政策ヘッド（次の手の確率分布）- 64マス + パス
        policy_head = tf.keras.layers.Dense(128, activation='relu')(features)
        policy_head = tf.keras.layers.Dense(65, activation='softmax', name='policy')(policy_head)

        # 価値ヘッド（勝率予測）
        value_head = tf.keras.layers.Dense(128, activation='relu')(features)
        value_head = tf.keras.layers.Dense(1, activation='tanh', name='value')(value_head)

        # モデルの構築
        model = tf.keras.Model(inputs=input_layer, outputs=[policy_head, value_head])

        # コンパイル
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error'
            }
        )

        return model

    def save(self, model_path: str) -> None:
        """モデルを保存する"""
        self.model.save(model_path)

    def _preprocess_board(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        盤面を3チャンネルの入力形式に変換する

        チャンネル0: 自分の石 (1または0)
        チャンネル1: 相手の石 (1または0)
        チャンネル2: 空きマス (1または0)
        """
        opponent = 3 - player

        # 3チャンネルの盤面を作成
        processed_board = np.zeros((8, 8, 3), dtype=np.float32)

        # 自分の石
        processed_board[:, :, 0] = (board == player).astype(np.float32)
        # 相手の石
        processed_board[:, :, 1] = (board == opponent).astype(np.float32)
        # 空きマス
        processed_board[:, :, 2] = (board == 0).astype(np.float32)

        return processed_board

    def predict(self, board: np.ndarray, player: int, valid_moves: List[Tuple[int, int]]) -> Tuple[Dict[str, int], float]:
        """
        現在の盤面から次の一手と勝率を予測する

        戻り値:
            - 次の手: {"row": int, "col": int} または None（パスの場合）
            - 勝率予測: float (0.0~1.0)
        """
        # 盤面の前処理
        processed_board = self._preprocess_board(board, player)

        # バッチ次元を追加
        input_data = np.expand_dims(processed_board, axis=0)

        # モデルの予測
        policy_pred, value_pred = self.model.predict(input_data, verbose=0)

        # 政策（行動確率）を取得
        policy = policy_pred[0]

        # 有効な手がない場合はパスを返す
        if not valid_moves:
            return None, (value_pred[0][0] + 1) / 2  # tanhの出力を0~1に変換

        # 有効な手の確率だけを抽出
        valid_moves_prob = {}
        for row, col in valid_moves:
            move_idx = row * 8 + col
            valid_moves_prob[(row, col)] = policy[move_idx]

        # 最も確率の高い手を選択
        best_move = max(valid_moves_prob.items(), key=lambda x: x[1])[0]

        # 勝率を0~1の範囲に変換（tanhの出力は-1~1）
        win_rate = (value_pred[0][0] + 1) / 2

        return {"row": best_move[0], "col": best_move[1]}, win_rate
