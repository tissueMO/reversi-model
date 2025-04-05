import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.trainer.self_play import SelfPlayManager
from src.trainer.visualization import VisualizationManager


class EvaluationManager(SelfPlayManager):
    """モデル評価を管理するクラス"""

    def _visualize_board(self, board, move_history=None):
        """可視化マネージャーを使用してボードを可視化する"""
        return VisualizationManager.visualize_board(board, move_history)

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
