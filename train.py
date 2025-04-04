import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import shutil
from datetime import datetime
from collections import deque
import gc
import psutil

from game import ReversiEnv
from model import ReversiModel


class SelfPlayTrainer:
    """自己対戦による強化学習のトレーナークラス"""

    def __init__(self, model=None, buffer_size=10000, log_dir="./logs"):
        """トレーナーを初期化する"""
        self.model = model if model else ReversiModel()
        self.env = ReversiEnv()
        self.buffer_size = buffer_size

        # 経験リプレイバッファ
        self.replay_buffer = deque(maxlen=buffer_size)

        # 学習の履歴
        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

        # 探索のパラメータ
        self.temperature = 1.0  # 探索温度

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

    @classmethod
    def load_from_model(cls, model_path, buffer_size=10000, log_dir="./logs"):
        """既存のモデルからトレーナーを初期化する"""
        print(f"モデル {model_path} から学習を再開します")
        # モデルをロード
        model = ReversiModel(model_path)
        # トレーナーを初期化
        return cls(model=model, buffer_size=buffer_size, log_dir=log_dir)

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

        # 早期停止用のコールバックを作成
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=3,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )

        # モデルの訓練
        history = self.model.model.fit(
            inputs,
            {'policy': policy_targets, 'value': value_targets},
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=[tensorboard_callback, early_stopping]
        )

        # 学習の履歴を保存
        for i in range(epochs):
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
                shutil.rmtree(item_path)
                print(f"- {item} を削除しました")

    def evaluate_model_strength(self, num_games=10, opponent_model=None, visualize=False):
        """モデルの強さを評価する（別のモデルと対戦する）"""
        wins = 0
        draws = 0
        losses = 0

        # 対戦相手がない場合は、ランダムに手を選択するプレイヤーとして扱う
        is_random_opponent = opponent_model is None

        for game_idx in tqdm(range(num_games), desc="評価対局"):
            # ゲームをリセット
            self.env.reset()

            # プレイヤーの割り当て: モデル=1, 対戦相手=2
            model_player = 1
            opponent_player = 2

            # ボードの状態を追跡
            game_states = []

            # 手番を記録する配列を追加
            move_history = np.zeros((8, 8), dtype=np.int32)
            move_count = 0

            while not self.env.done:
                player = self.env.current_player
                board = self.env.board.copy()

                # 現在の盤面状態を保存（可視化用）
                if visualize:
                    game_states.append(board.copy())

                # 有効な手の取得
                valid_moves = self.env.get_valid_moves(player)

                if not valid_moves:
                    # 有効な手がない場合はパス
                    self.env.current_player = 3 - player
                    continue

                # プレイヤーに応じて異なる行動選択方法を使用
                if player == model_player:
                    # 評価対象のモデルを使用
                    action_probs, valid_indices = self._get_action_probs(
                        board, player, valid_moves, temperature=0.1)
                    action_idx = np.argmax(action_probs)  # 最も確率の高い手を選ぶ
                    row, col = valid_indices[action_idx]
                else:
                    # 対戦相手の行動選択
                    if is_random_opponent:
                        # ランダムに選択
                        move_idx = np.random.choice(len(valid_moves))
                        row, col = valid_moves[move_idx]
                    else:
                        # 対戦相手のモデルを使用
                        action_probs, valid_indices = opponent_model._get_action_probs(
                            board, player, valid_moves, temperature=0.1)
                        action_idx = np.argmax(action_probs)
                        row, col = valid_indices[action_idx]

                # 手番をカウントアップして記録
                move_count += 1
                move_history[row, col] = move_count

                # 選択した手を実行
                self.env.make_move(row, col)

            # 結果を記録
            winner = self.env.get_winner()
            if winner == 0:  # 引き分け
                draws += 1
            elif winner == model_player:  # 勝ち
                wins += 1
            else:  # 負け
                losses += 1

            # ゲーム結果をTensorBoardに記録
            with self.summary_writer.as_default():
                tf.summary.text(
                    f'game_{game_idx}',
                    f'結果: {"引き分け" if winner == 0 else ("勝ち" if winner == model_player else "負け")} ' +
                    f'スコア: {np.sum(self.env.board == model_player)}-{np.sum(self.env.board == opponent_player)}',
                    step=game_idx
                )

                # ボード状態の可視化（最終盤面）
                if visualize and len(game_states) > 0:
                    # 最終盤面の可視化（画像として）- 手番の順序を含める
                    final_board = self._visualize_board(
                        self.env.board, move_history)
                    tf.summary.image(
                        f"game_{game_idx}/final_board", final_board, step=game_idx)

        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games

        # 評価結果をTensorBoardに記録
        with self.summary_writer.as_default():
            tf.summary.scalar('evaluation/win_rate', win_rate, step=0)
            tf.summary.scalar('evaluation/draw_rate', draw_rate, step=0)
            tf.summary.scalar('evaluation/loss_rate', loss_rate, step=0)

        return {
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate
        }

    def _visualize_board(self, board, move_history=None):
        """
        ボードを可視化し、画像として返す

        Args:
            board: 盤面情報（0=空きマス, 1=プレイヤー1の石, 2=プレイヤー2の石）
            move_history: 各マスが何手目に打たれたかの情報（任意）
        """
        # 拡大倍率を定義
        scale = 32  # 16から32に拡大して解像度を上げる

        # 8x8の盤面を3チャンネルのRGB画像に変換
        vis_board = np.zeros((8 * scale, 8 * scale, 3), dtype=np.uint8)

        # マス目の線を描画
        for i in range(9):
            # 縦線
            vis_board[:, i * scale - 1 if i > 0 else 0: i *
                      scale + 1 if i < 8 else -1, :] = [50, 50, 50]
            # 横線
            vis_board[i * scale - 1 if i > 0 else 0: i *
                      scale + 1 if i < 8 else -1, :, :] = [50, 50, 50]

        # 各マスの背景を緑色に
        for row in range(8):
            for col in range(8):
                r_start, r_end = row * scale + 1, (row + 1) * scale - 1
                c_start, c_end = col * scale + 1, (col + 1) * scale - 1
                vis_board[r_start:r_end, c_start:c_end] = [0, 100, 0]

        # 石を描画
        for row in range(8):
            for col in range(8):
                r_center, c_center = row * scale + scale//2, col * scale + scale//2
                radius = scale//2 - 2

                if board[row, col] == 1:  # プレイヤー1の石（黒）
                    self._draw_circle(vis_board, r_center,
                                      c_center, radius, [0, 0, 0])
                elif board[row, col] == 2:  # プレイヤー2の石（白）
                    self._draw_circle(vis_board, r_center,
                                      c_center, radius, [255, 255, 255])

        # 手番の順序を数字で表示
        if move_history is not None:
            for row in range(8):
                for col in range(8):
                    move_number = move_history[row, col]
                    if move_number > 0:
                        r_center, c_center = row * scale + scale//2, col * scale + scale//2
                        # 数字の表示位置を調整
                        if board[row, col] == 1:  # 黒石上の数字は白色
                            color = [255, 255, 255]
                        elif board[row, col] == 2:  # 白石上の数字は黒色
                            color = [0, 0, 0]
                        else:  # ありえないはずだが念のため
                            continue

                        # 数字のサイズを大きくする（石の半径の70%程度）
                        size = max(3, int(radius * 0.7))
                        self._draw_number(vis_board, r_center,
                                          c_center, move_number, color, size)

        # バッチ次元を追加（TensorBoardのimage summaryのため）
        return np.expand_dims(vis_board, axis=0)

    def _draw_circle(self, image, center_r, center_c, radius, color):
        """
        イメージ上に円を描く簡易関数
        """
        rows, cols = image.shape[:2]

        # 円の範囲内のピクセルを設定
        for r in range(max(0, center_r - radius), min(rows, center_r + radius + 1)):
            for c in range(max(0, center_c - radius), min(cols, center_c + radius + 1)):
                # 中心からの距離を計算
                if ((r - center_r) ** 2 + (c - center_c) ** 2) <= radius ** 2:
                    image[r, c] = color

    def _draw_number(self, image, center_r, center_c, number, color, size):
        """
        イメージ上に数字を描く関数（0〜99の数値に対応）

        各数字を個別に定義して明確に描画します
        """
        # 数字を文字列に変換
        num_str = str(number)

        # 数字の太さを増加（より見やすく）
        thickness = max(2, size // 4)  # 太さを増加

        # 中心から少しずらして数字を描画（複数桁の場合も考慮）
        offset = len(num_str) * size // 2  # オフセットを大きくしてスペースを確保

        for i, digit in enumerate(num_str):
            # 数字の位置を調整（複数桁の場合は横に並べる）
            digit_center_c = center_c - offset + (i * size) + size // 2

            # 各数字のパターンに基づいて描画
            self._draw_digit(image, digit, center_r,
                             digit_center_c, color, size)

    def _draw_digit(self, image, digit, center_r, center_c, color, size):
        """
        単一の数字（0〜9）を描画する補助関数
        """
        # セグメントの定義（7セグメントディスプレイ風）
        segments = {
            # 各セグメントの有無を定義（上、右上、右下、下、左下、左上、中央）
            '0': (True,  True,  True,  True,  True,  True,  False),
            '1': (False, True,  True,  False, False, False, False),
            '2': (True,  True,  False, True,  True,  False, True),
            '3': (True,  True,  True,  True,  False, False, True),
            '4': (False, True,  True,  False, False, True,  True),
            '5': (True,  False, True,  True,  False, True,  True),
            '6': (True,  False, True,  True,  True,  True,  True),
            '7': (True,  True,  True,  False, False, False, False),
            '8': (True,  True,  True,  True,  True,  True,  True),
            '9': (True,  True,  True,  True,  False, True,  True)
        }

        # サイズの調整（数字全体のサイズはsize）
        half_width = size // 2
        half_height = size // 2
        thickness = max(2, size // 4)  # 数字の線の太さを増加

        # セグメントの座標を取得
        segment_points = self._get_segment_points(
            center_r, center_c, half_height, half_width)

        # 対応するセグメントを描画
        segment_patterns = segments.get(digit, segments['0'])  # 未定義の文字は0として扱う

        for i, active in enumerate(segment_patterns):
            if active:
                # セグメントの描画（セグメントごとに異なる描画処理）
                start_point, end_point = segment_points[i]
                if i in [0, 3, 6]:  # 水平なセグメント（上、下、中央）
                    self._draw_horizontal_segment(
                        image, start_point, end_point, color, thickness)
                else:  # 垂直なセグメント
                    self._draw_vertical_segment(
                        image, start_point, end_point, color, thickness)

    def _get_segment_points(self, center_r, center_c, half_height, half_width):
        """セグメント表示の各点を計算"""
        # 各セグメントの始点と終点を定義
        # (r_start, c_start), (r_end, c_end) の形式
        top = ((center_r - half_height, center_c - half_width),
               (center_r - half_height, center_c + half_width))
        top_right = ((center_r - half_height, center_c + half_width),
                     (center_r, center_c + half_width))
        bottom_right = ((center_r, center_c + half_width),
                        (center_r + half_height, center_c + half_width))
        bottom = ((center_r + half_height, center_c - half_width),
                  (center_r + half_height, center_c + half_width))
        bottom_left = ((center_r, center_c - half_width),
                       (center_r + half_height, center_c - half_width))
        top_left = ((center_r - half_height, center_c - half_width),
                    (center_r, center_c - half_width))
        middle = ((center_r, center_c - half_width),
                  (center_r, center_c + half_width))

        return [top, top_right, bottom_right, bottom, bottom_left, top_left, middle]

    def _draw_horizontal_segment(self, image, start_point, end_point, color, thickness):
        """水平なセグメントを描画"""
        r, c_start = start_point
        _, c_end = end_point

        for c in range(c_start, c_end + 1):
            for t in range(-thickness, thickness + 1):  # 太さを増加
                if 0 <= r + t < image.shape[0] and 0 <= c < image.shape[1]:
                    image[r + t, c] = color

    def _draw_vertical_segment(self, image, start_point, end_point, color, thickness):
        """垂直なセグメントを描画"""
        r_start, c = start_point
        r_end, _ = end_point

        for r in range(r_start, r_end + 1):
            for t in range(-thickness, thickness + 1):  # 太さを増加
                if 0 <= r < image.shape[0] and 0 <= c + t < image.shape[1]:
                    image[r, c + t] = color


def main():
    parser = argparse.ArgumentParser(description='自己対戦によるリバーシAIモデルの学習')
    parser.add_argument('--games', type=int, default=100, help='自己対戦するゲーム数')
    parser.add_argument('--iterations', type=int,
                        default=10, help='学習のイテレーション回数')
    parser.add_argument('--epochs', type=int, default=10,
                        help='各イテレーションでのエポック数')
    parser.add_argument('--batch-size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--temperature', type=float,
                        default=1.0, help='行動選択の温度パラメータ（高いほど探索的に）')
    parser.add_argument('--buffer-size', type=int,
                        default=10000, help='リプレイバッファのサイズ')
    parser.add_argument('--output-dir', type=str,
                        default='./output', help='モデルの保存先ディレクトリ')
    parser.add_argument('--log-dir', type=str,
                        default='./logs', help='TensorBoardログの保存先ディレクトリ')
    parser.add_argument('--evaluate', action='store_true',
                        help='学習後にモデルの強さを評価する')
    parser.add_argument('--visualize', action='store_true',
                        help='評価中のボード状態を可視化する')
    parser.add_argument('--save-interim', action='store_true',
                        help='中間モデルを保存するかどうか（デフォルトは保存しない）')
    parser.add_argument('--load-model', type=str, help='学習を再開する既存モデルのパス')
    parser.add_argument('--start-iteration', type=int,
                        default=0, help='学習を再開する場合の開始イテレーション番号')
    parser.add_argument('--memory-growth',
                        action='store_true', help='GPU メモリの動的確保を有効にする')
    parser.add_argument('--memory-log-interval', type=int,
                        default=10, help='メモリ使用状況をログに記録する間隔')

    args = parser.parse_args()

    # GPUメモリの設定（OOM対策）
    if args.memory_growth:
        print("GPU メモリの動的確保を有効化しています...")
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"{len(physical_devices)} 個のGPUでメモリ動的確保が有効化されました")
            else:
                print("利用可能なGPUがありません")
        except Exception as e:
            print(f"GPUメモリ設定でエラーが発生しました: {str(e)}")

    # TensorFlowのログレベルを設定（冗長なログを削減）
    tf.get_logger().setLevel('ERROR')

    # メモリ使用量の初期状態をログ出力
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)
    print(f"初期メモリ使用量: {initial_memory:.2f} MB")

    # 出力ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # トレーナーの初期化（既存モデルがある場合はロード）
    if args.load_model:
        trainer = SelfPlayTrainer.load_from_model(
            model_path=args.load_model,
            buffer_size=args.buffer_size,
            log_dir=args.log_dir
        )
        start_iteration = args.start_iteration
    else:
        trainer = SelfPlayTrainer(
            buffer_size=args.buffer_size, log_dir=args.log_dir)
        start_iteration = 0

    # 学習ループ
    for i in range(start_iteration, start_iteration + args.iterations):
        print(f"\nイテレーション {i+1}/{start_iteration + args.iterations}")

        # 自己対戦でデータ生成
        trainer.generate_self_play_data(
            num_games=args.games, temperature=args.temperature)

        # モデルの訓練
        losses = trainer.train(batch_size=args.batch_size,
                               epochs=args.epochs, iteration=i)

        print(
            f"損失 - ポリシー: {losses['policy_loss']:.4f}, 価値: {losses['value_loss']:.4f}, 合計: {losses['total_loss']:.4f}")

        # 中間モデルを保存するオプションが指定されている場合のみ中間モデルを保存
        if args.save_interim:
            trainer.save_model(path=os.path.join(
                output_dir, f"model_iter_{i+1}"))

    # 最終モデルを保存
    trainer.save_model(path=os.path.join(output_dir, "final_model"))

    # もし中間モデルを保存していた場合は削除
    if args.save_interim:
        trainer.remove_interim_models(output_dir)

    # モデルの評価
    if args.evaluate:
        print("\nモデルの強さを評価中...")
        evaluation_results = trainer.evaluate_model_strength(
            num_games=args.games, visualize=args.visualize)
        print(f"評価結果: 勝ち: {evaluation_results['wins']}回, 引き分け: {evaluation_results['draws']}回, " +
              f"負け: {evaluation_results['losses']}回, 勝率: {evaluation_results['win_rate']:.2%}")

    print(f"\n学習が完了しました。モデルが {output_dir}/final_model に保存されました")
    if args.save_interim:
        print("すべての中間モデルは削除されました")
    print(f"TensorBoardログは {trainer.log_dir} に保存されました")
    print(f"学習の進捗をTensorBoardで確認するには: tensorboard --logdir={args.log_dir}")
    print("その後、ブラウザで http://localhost:6006 を開いてください")


if __name__ == "__main__":
    main()
