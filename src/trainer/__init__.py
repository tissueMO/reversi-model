import os
from datetime import datetime

from src.game.reversi import ReversiEnv
from src.model.model import ReversiModel
from src.trainer.evaluation import EvaluationManager


class ReversiTrainer(EvaluationManager):
    """リバーシゲームの強化学習トレーナー

    このクラスは分割された機能を統合し、トレーニングプロセス全体を管理します。
    """

    def __init__(self, model=None, buffer_size=10000, log_dir="./output/logs"):
        """トレーナーを初期化する"""
        # 依存オブジェクトの初期化
        self.env = ReversiEnv()
        model = model if model else ReversiModel()
        
        # 親クラスの初期化
        super().__init__(model=model, env=self.env, buffer_size=buffer_size, log_dir=log_dir)

    @classmethod
    def load_from_model(cls, model_path, buffer_size=10000, log_dir="./output/logs"):
        """既存のモデルからトレーナーを初期化する"""
        print(f"モデル {model_path} から学習を再開します")
        # モデルをロード
        model = ReversiModel(model_path)
        # トレーナーを初期化
        return cls(model=model, buffer_size=buffer_size, log_dir=log_dir)