import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import gc
import psutil


class BaseTrainer:
    """強化学習トレーナーの基本クラス"""

    def __init__(self, model=None, buffer_size=10000, log_dir="./output/logs"):
        """トレーナーを初期化する"""
        self.model = model
        self.buffer_size = buffer_size

        # 学習の履歴
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

        # TensorBoardのログディレクトリ
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        # メモリ使用状況の監視のための初期設定
        self.memory_logs_enabled = True
        self.memory_check_counter = 0
        self.memory_check_interval = 10  # 10回ごとにメモリ使用状況をログに記録

    def log_memory_usage(self):
        """現在のメモリ使用状況をログに記録する"""
        if not self.memory_logs_enabled:
            return

        # 現在のプロセスのメモリ使用状況を取得
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # MB単位で表示
        memory_mb = memory_info.rss / 1024 / 1024

        # TensorBoardにメモリ使用量を記録
        with self.summary_writer.as_default():
            tf.summary.scalar('memory_usage_mb', memory_mb,
                            step=self.memory_check_counter)

        # 標準出力にもログを表示
        print(f"メモリ使用量: {memory_mb:.2f} MB")
        self.memory_check_counter += 1

    def clear_tensorflow_session(self):
        """TensorFlowのセッションをクリアしてメモリを解放する"""
        # Kerasバックエンドのセッションをクリア
        tf.keras.backend.clear_session()
        # 明示的にガベージコレクションを実行
        gc.collect()

    def save_model(self, path="./trained_model"):
        """学習したモデルを保存する"""
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/reversi_model")

    def remove_interim_models(self, output_dir):
        """中間モデルを削除する"""
        # 削除する前に確認メッセージを表示
        print("中間モデルを削除中...")

        # ディレクトリ内の項目を確認
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)

            # model_iter_X形式のディレクトリを探す
            if os.path.isdir(item_path) and item.startswith("model_iter_"):
                # ディレクトリを削除
                import shutil
                shutil.rmtree(item_path)
                print(f"- {item} を削除しました")