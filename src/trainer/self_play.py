import numpy as np
from tqdm import tqdm

from src.trainer.training import TrainingManager


class SelfPlayManager(TrainingManager):
    """自己対戦による学習データ生成を管理するクラス"""

    def __init__(self, model=None, env=None, buffer_size=10000, log_dir="./output/logs"):
        """自己対戦マネージャーを初期化する"""
        super().__init__(model=model, buffer_size=buffer_size, log_dir=log_dir)

        self.env = env
        # 探索のパラメータ
        self.temperature = 1.0  # 探索温度

    def _get_action_probs(self, board, player, valid_moves, temperature=1.0):
        """
        モデルの予測から、有効な手の確率分布を取得する
        温度が低いほど、最善手を選びやすくなる
        """
        # 盤面の前処理
        processed_board = self._preprocess_board(board, player)

        # バッチ次元を追加
        input_data = np.expand_dims(processed_board, axis=0)

        # モデルの予測
        policy_pred, _ = self.model.model.predict(input_data, verbose=0)

        # 政策（行動確率）を取得
        policy = policy_pred[0]

        # 有効な手だけの確率を抽出
        valid_probs = []
        valid_indices = []

        for row, col in valid_moves:
            move_idx = row * 8 + col
            valid_probs.append(policy[move_idx])
            valid_indices.append((row, col))

        # 温度でスケーリング
        if temperature == 0:
            # 温度が0の場合は最善手のみを選択（確定的）
            best_idx = np.argmax(valid_probs)
            action_probs = np.zeros(len(valid_probs))
            action_probs[best_idx] = 1.0
        else:
            # 負の値や極端に小さい値を防ぐためにクリップ
            valid_probs = np.array(valid_probs)
            valid_probs = np.clip(valid_probs, 1e-10, None)  # 小さな正の値でクリップ

            # Boltzmann分布でスケーリング
            valid_probs = valid_probs ** (1 / temperature)

            # NaNや無限大をチェックして置き換え
            valid_probs = np.nan_to_num(
                valid_probs, nan=1e-10, posinf=1.0, neginf=1e-10)

            # 合計が0になるのを防ぐ
            if np.sum(valid_probs) <= 0:
                # すべての手に均等な確率を割り当て
                action_probs = np.ones(len(valid_probs)) / len(valid_probs)
            else:
                action_probs = valid_probs / np.sum(valid_probs)

        return action_probs, valid_indices

    def _play_game(self, temperature=1.0):
        """1ゲームを自己対戦でプレイし、訓練データを生成する"""
        self.env.reset()
        game_history = []

        while not self.env.done:
            player = self.env.current_player
            board = self.env.board.copy()

            # 有効な手の取得
            valid_moves = self.env.get_valid_moves(player)

            if not valid_moves:
                # 有効な手がない場合はパス
                action = None
                self.env.current_player = 3 - player
                continue

            # モデルから行動確率を取得
            action_probs, valid_indices = self._get_action_probs(
                board, player, valid_moves, temperature)

            # 行動をサンプリング
            action_idx = np.random.choice(len(valid_indices), p=action_probs)
            row, col = valid_indices[action_idx]

            # 学習用のデータを保存
            # one-hot形式の行動ベクトル
            action_vector = np.zeros(65)  # 64マス + パス
            action_vector[row * 8 + col] = 1

            game_history.append({
                'board': board.copy(),  # 明示的にコピーすることでメモリリークを防ぐ
                'player': player,
                'policy': action_vector,
                'value': None  # ゲーム終了後に更新
            })

            # 選択した手を実行
            next_board, reward, done, info = self.env.make_move(row, col)

        # ゲーム終了後、各ステップの価値（勝率）を更新
        winner = self.env.get_winner()

        for i in range(len(game_history)):
            player = game_history[i]['player']

            if winner == 0:  # 引き分け
                game_history[i]['value'] = 0.0
            elif winner == player:  # 勝ち
                game_history[i]['value'] = 1.0
            else:  # 負け
                game_history[i]['value'] = -1.0

        return game_history

    def generate_self_play_data(self, num_games=100, temperature=1.0):
        """複数ゲームの自己対戦を行い、データを生成する"""
        for i in tqdm(range(num_games), desc="自己対戦ゲーム"):
            game_history = self._play_game(temperature)
            self.replay_buffer.extend(game_history)

            # 定期的にメモリ使用状況をログに記録し、不要なメモリを解放
            if i % self.memory_check_interval == 0:
                self.log_memory_usage()
                self.clear_tensorflow_session()
