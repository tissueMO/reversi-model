import numpy as np
import tensorflow as tf
from collections import deque

from src.trainer.base_trainer import BaseTrainer


class TrainingManager(BaseTrainer):
    """モデル学習に関する機能を提供するクラス"""

    def __init__(self, model=None, buffer_size=10000, log_dir="./output/logs"):
        """トレーニングマネージャーを初期化する"""
        super().__init__(model=model, buffer_size=buffer_size, log_dir=log_dir)

        # 経験リプレイバッファ
        self.replay_buffer = deque(maxlen=buffer_size)

    def _preprocess_board(self, board, player):
        """モデル入力用に盤面を前処理する"""
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

    def train(self, batch_size=32, epochs=10, iteration=0):
        """リプレイバッファからサンプリングしてモデルを学習させる"""
        if len(self.replay_buffer) < batch_size:
            print(f"学習に必要なデータが不足しています。少なくとも {batch_size} サンプルが必要です。")
            return

        # 開始時のメモリ使用状況を記録
        self.log_memory_usage()

        # データの前処理
        inputs = []
        policy_targets = []
        value_targets = []

        # ランダムサンプリング
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size)

        for idx in batch_indices:
            data = self.replay_buffer[idx]
            board = data['board']
            player = data['player']

            # 盤面の前処理
            processed_board = self._preprocess_board(board, player)

            inputs.append(processed_board)
            policy_targets.append(data['policy'])
            value_targets.append(data['value'])

        # NumPy配列に変換
        inputs = np.array(inputs)
        policy_targets = np.array(policy_targets)
        value_targets = np.array(value_targets).reshape(-1, 1)

        # TensorBoard用のコールバックを作成
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )

        # モデルの訓練
        history = self.model.model.fit(
            inputs,
            {'policy': policy_targets, 'value': value_targets},
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[tensorboard_callback]
        )

        # 学習の履歴を保存（早期停止に対応）
        actual_epochs = len(history.history['loss'])
        for i in range(actual_epochs):
            self.policy_losses.append(history.history['policy_loss'][i])
            self.value_losses.append(history.history['value_loss'][i])
            self.total_losses.append(history.history['loss'][i])

            # TensorBoardにメトリクスを記録
            step = iteration * epochs + i
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    'policy_loss', history.history['policy_loss'][i], step=step)
                tf.summary.scalar(
                    'value_loss', history.history['value_loss'][i], step=step)
                tf.summary.scalar(
                    'total_loss', history.history['loss'][i], step=step)

        # 訓練後のメモリクリーンアップ
        # NumPy配列の明示的な解放
        del inputs
        del policy_targets
        del value_targets
        self.clear_tensorflow_session()
        self.log_memory_usage()

        return {
            'policy_loss': np.mean(history.history['policy_loss']),
            'value_loss': np.mean(history.history['value_loss']),
            'total_loss': np.mean(history.history['loss'])
        }
