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

    def _get_position_weights(self):
        """盤面の各位置の重み付けを定義する"""
        # 8x8の盤面における各位置の重み
        weights = np.array([
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -30, 1, 1, 1, 1, -30, -20],
            [10, 1, 5, 2, 2, 5, 1, 10],
            [5, 1, 2, 1, 1, 2, 1, 5],
            [5, 1, 2, 1, 1, 2, 1, 5],
            [10, 1, 5, 2, 2, 5, 1, 10],
            [-20, -30, 1, 1, 1, 1, -30, -20],
            [100, -20, 10, 5, 5, 10, -20, 100]
        ])
        return weights

    def _evaluate_board(self, board, player):
        """盤面の評価値を計算する

        Args:
            board: 現在の盤面
            player: 評価するプレイヤー

        Returns:
            評価値（高いほど有利）
        """
        opponent = 3 - player
        weights = self._get_position_weights()

        # 残りのマス数を計算（空きマスの数）
        empty_count = np.sum(board == 0)

        # 終盤（残り16マス以下）かどうか
        is_endgame = empty_count <= 16

        if is_endgame:
            # 終盤では単純に石数の差を最大化
            player_count = np.sum(board == player)
            opponent_count = np.sum(board == opponent)
            return player_count - opponent_count

        # 位置による重み付けスコア
        position_score = 0
        for i in range(8):
            for j in range(8):
                if board[i, j] == player:
                    position_score += weights[i, j]
                elif board[i, j] == opponent:
                    position_score -= weights[i, j]

        # 相手の有効手の数を計算（少ないほど良い）
        valid_moves_opponent = self.env.get_valid_moves(opponent)
        mobility_score = -len(valid_moves_opponent) * 2  # 移動度に対する重み付け

        # 自分の有効手の数も加味
        valid_moves_player = self.env.get_valid_moves(player)
        mobility_score += len(valid_moves_player)

        # 総合評価
        total_score = position_score + mobility_score

        return total_score

    def _select_best_move(self, board, player, valid_moves):
        """最も評価値の高い手を選択する"""
        best_score = float('-inf')
        best_move_idx = 0

        # 盤面の一時的なコピーを作成
        temp_env = self.env.__class__()

        for i, (row, col) in enumerate(valid_moves):
            # 一時的な盤面を現在の状態にリセット
            temp_env.board = board.copy()
            temp_env.current_player = player

            # 手を打ってみる
            temp_env.make_move(row, col)

            # 盤面を評価
            score = self._evaluate_board(temp_env.board, player)

            # より良いスコアが見つかれば更新
            if score > best_score:
                best_score = score
                best_move_idx = i

        # 最良の手のインデックスを返す
        return best_move_idx

    def evaluate_model_strength(self, num_games=10, opponent_model=None, visualize=False):
        """モデルの強さを評価する（別のモデルと対戦する）"""
        wins = 0
        draws = 0
        losses = 0

        # 対戦相手がない場合は、ロジックベースのAIを使用
        is_logic_based_opponent = opponent_model is None

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
                    if is_logic_based_opponent:
                        # ロジックベースのAIによる選択
                        move_idx = self._select_best_move(board, player, valid_moves)
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
