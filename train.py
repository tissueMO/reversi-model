import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
import argparse
import shutil
from datetime import datetime
from collections import deque

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
            # Boltzmann分布でスケーリング
            valid_probs = np.array(valid_probs) ** (1 / temperature)
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
            action_probs, valid_indices = self._get_action_probs(board, player, valid_moves, temperature)

            # 行動をサンプリング
            action_idx = np.random.choice(len(valid_indices), p=action_probs)
            row, col = valid_indices[action_idx]

            # 学習用のデータを保存
            # one-hot形式の行動ベクトル
            action_vector = np.zeros(65)  # 64マス + パス
            action_vector[row * 8 + col] = 1

            game_history.append({
                'board': board,
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
        for _ in tqdm(range(num_games), desc="自己対戦ゲーム"):
            game_history = self._play_game(temperature)
            self.replay_buffer.extend(game_history)

    def train(self, batch_size=32, epochs=10, iteration=0):
        """リプレイバッファからサンプリングしてモデルを学習させる"""
        if len(self.replay_buffer) < batch_size:
            print(f"学習に必要なデータが不足しています。少なくとも {batch_size} サンプルが必要です。")
            return

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

        # 学習の履歴を保存
        for i in range(epochs):
            self.policy_losses.append(history.history['policy_loss'][i])
            self.value_losses.append(history.history['value_loss'][i])
            self.total_losses.append(history.history['loss'][i])

            # TensorBoardにメトリクスを記録
            step = iteration * epochs + i
            with self.summary_writer.as_default():
                tf.summary.scalar('policy_loss', history.history['policy_loss'][i], step=step)
                tf.summary.scalar('value_loss', history.history['value_loss'][i], step=step)
                tf.summary.scalar('total_loss', history.history['loss'][i], step=step)

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
                    action_probs, valid_indices = self._get_action_probs(board, player, valid_moves, temperature=0.1)
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
                        action_probs, valid_indices = opponent_model._get_action_probs(board, player, valid_moves, temperature=0.1)
                        action_idx = np.argmax(action_probs)
                        row, col = valid_indices[action_idx]

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
                    # 最終盤面の可視化（画像として）
                    final_board = self._visualize_board(self.env.board)
                    tf.summary.image(f"game_{game_idx}/final_board", final_board, step=game_idx)

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

    def _visualize_board(self, board):
        """ボードを可視化し、画像として返す"""
        # 8x8の盤面を3チャンネルのRGB画像に変換
        vis_board = np.zeros((8, 8, 3), dtype=np.uint8)

        # 背景色（緑）
        vis_board[:, :] = [0, 100, 0]

        # プレイヤー1の石（黒）
        vis_board[board == 1] = [0, 0, 0]

        # プレイヤー2の石（白）
        vis_board[board == 2] = [255, 255, 255]

        # 1ピクセル = 1セルではわかりにくいので、拡大する
        scale = 10
        large_board = np.kron(vis_board, np.ones((scale, scale, 1), dtype=np.uint8))

        # バッチ次元を追加（TensorBoardのimage summaryのため）
        return np.expand_dims(large_board, axis=0)


def main():
    parser = argparse.ArgumentParser(description='自己対戦によるリバーシAIモデルの学習')
    parser.add_argument('--games', type=int, default=100, help='自己対戦するゲーム数')
    parser.add_argument('--iterations', type=int, default=10, help='学習のイテレーション回数')
    parser.add_argument('--epochs', type=int, default=10, help='各イテレーションでのエポック数')
    parser.add_argument('--batch-size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--temperature', type=float, default=1.0, help='行動選択の温度パラメータ（高いほど探索的に）')
    parser.add_argument('--buffer-size', type=int, default=10000, help='リプレイバッファのサイズ')
    parser.add_argument('--output-dir', type=str, default='./output', help='モデルの保存先ディレクトリ')
    parser.add_argument('--log-dir', type=str, default='./logs', help='TensorBoardログの保存先ディレクトリ')
    parser.add_argument('--evaluate', action='store_true', help='学習後にモデルの強さを評価する')
    parser.add_argument('--visualize', action='store_true', help='評価中のボード状態を可視化する')
    parser.add_argument('--save-interim', action='store_true', help='中間モデルを保存するかどうか（デフォルトは保存しない）')
    parser.add_argument('--load-model', type=str, help='学習を再開する既存モデルのパス')
    parser.add_argument('--start-iteration', type=int, default=0, help='学習を再開する場合の開始イテレーション番号')

    args = parser.parse_args()

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
        trainer = SelfPlayTrainer(buffer_size=args.buffer_size, log_dir=args.log_dir)
        start_iteration = 0

    # 学習ループ
    for i in range(start_iteration, start_iteration + args.iterations):
        print(f"\nイテレーション {i+1}/{start_iteration + args.iterations}")

        # 自己対戦でデータ生成
        trainer.generate_self_play_data(num_games=args.games, temperature=args.temperature)

        # モデルの訓練
        losses = trainer.train(batch_size=args.batch_size, epochs=args.epochs, iteration=i)

        print(f"損失 - ポリシー: {losses['policy_loss']:.4f}, 価値: {losses['value_loss']:.4f}, 合計: {losses['total_loss']:.4f}")

        # 中間モデルを保存するオプションが指定されている場合のみ中間モデルを保存
        if args.save_interim:
            trainer.save_model(path=os.path.join(output_dir, f"model_iter_{i+1}"))

    # 最終モデルを保存
    trainer.save_model(path=os.path.join(output_dir, "final_model"))

    # もし中間モデルを保存していた場合は削除
    if args.save_interim:
        trainer.remove_interim_models(output_dir)

    # モデルの評価
    if args.evaluate:
        print("\nモデルの強さを評価中...")
        evaluation_results = trainer.evaluate_model_strength(num_games=args.games, visualize=args.visualize)
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
