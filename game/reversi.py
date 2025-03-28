import numpy as np
from typing import Tuple, List, Dict, Optional, Union

class ReversiEnv:
    """リバーシゲームの環境クラス"""

    def __init__(self):
        """ゲーム環境を初期化する"""
        # 8x8の盤面を作成 (0=空きマス, 1=プレイヤー1の石, 2=プレイヤー2の石)
        self.board = np.zeros((8, 8), dtype=np.int8)

        # 初期配置
        self.board[3, 3] = 1
        self.board[4, 4] = 1
        self.board[3, 4] = 2
        self.board[4, 3] = 2

        # 現在のプレイヤー (1または2)
        self.current_player = 1

        # ゲーム終了フラグ
        self.done = False

        # 方向ベクトル (8方向)
        self.directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    def reset(self) -> np.ndarray:
        """ゲームをリセットし、初期状態の盤面を返す"""
        self.__init__()
        return self.board.copy()

    def is_valid_move(self, row: int, col: int, player: int) -> bool:
        """指定した位置が有効な手かどうか確認する"""
        # すでに石が置かれている場合は無効
        if self.board[row, col] != 0:
            return False

        # 8方向のいずれかで相手の石を挟めるかチェック
        for dr, dc in self.directions:
            r, c = row + dr, col + dc
            # 挟める石があるかのフラグ
            has_opponent_between = False

            # 盤面の範囲内で相手の石が続く限りループ
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == 3 - player:
                r += dr
                c += dc
                has_opponent_between = True

            # 相手の石を挟んで自分の石があれば有効な手
            if has_opponent_between and 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
                return True

        return False

    def get_valid_moves(self, player: int) -> List[Tuple[int, int]]:
        """プレイヤーの有効な手のリストを取得する"""
        valid_moves = []
        for row in range(8):
            for col in range(8):
                if self.is_valid_move(row, col, player):
                    valid_moves.append((row, col))
        return valid_moves

    def make_move(self, row: int, col: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        指定した位置に現在のプレイヤーの石を置き、盤面を更新する

        戻り値:
            - 更新された盤面
            - 報酬 (ゲーム終了時のみ: 勝利=1.0, 敗北=-1.0, 引き分け=0.0, それ以外は0.0)
            - ゲーム終了フラグ
            - 追加情報
        """
        if not self.is_valid_move(row, col, self.current_player):
            # 無効な手の場合、現在の状態を返す（実際の実装ではエラー処理が必要）
            return self.board.copy(), 0.0, self.done, {}

        # 石を置く
        self.board[row, col] = self.current_player

        # 8方向をチェックして石をひっくり返す
        for dr, dc in self.directions:
            self._flip_stones(row, col, dr, dc)

        # プレイヤーの切り替え
        self.current_player = 3 - self.current_player

        # 次のプレイヤーに有効な手があるかチェック
        valid_moves = self.get_valid_moves(self.current_player)

        # 有効な手がない場合
        if not valid_moves:
            # 相手（元のプレイヤー）に戻して、そのプレイヤーに有効な手があるかチェック
            self.current_player = 3 - self.current_player
            valid_moves = self.get_valid_moves(self.current_player)

            # 両プレイヤーとも有効な手がない場合、ゲーム終了
            if not valid_moves:
                self.done = True

                # 勝敗を判定
                p1_stones = np.sum(self.board == 1)
                p2_stones = np.sum(self.board == 2)

                reward = 0.0  # デフォルトは引き分け

                if p1_stones > p2_stones:
                    reward = 1.0 if self.current_player == 1 else -1.0
                elif p2_stones > p1_stones:
                    reward = 1.0 if self.current_player == 2 else -1.0

                info = {
                    "player1_stones": p1_stones,
                    "player2_stones": p2_stones,
                    "winner": 1 if p1_stones > p2_stones else (2 if p2_stones > p1_stones else 0)
                }

                return self.board.copy(), reward, self.done, info

        # ゲームが続行する場合
        return self.board.copy(), 0.0, self.done, {}

    def _flip_stones(self, row: int, col: int, dr: int, dc: int) -> None:
        """指定した方向に石をひっくり返す"""
        player = self.board[row, col]
        opponent = 3 - player

        # 相手の石を探索
        r, c = row + dr, col + dc
        stones_to_flip = []

        while 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == opponent:
            stones_to_flip.append((r, c))
            r += dr
            c += dc

        # 自分の石で挟めた場合、間の石をひっくり返す
        if 0 <= r < 8 and 0 <= c < 8 and self.board[r, c] == player:
            for flip_r, flip_c in stones_to_flip:
                self.board[flip_r, flip_c] = player

    def get_winner(self) -> int:
        """勝者を返す (0=引き分け, 1=プレイヤー1の勝ち, 2=プレイヤー2の勝ち)"""
        if not self.done:
            return -1  # ゲームが終了していない

        p1_stones = np.sum(self.board == 1)
        p2_stones = np.sum(self.board == 2)

        if p1_stones > p2_stones:
            return 1
        elif p2_stones > p1_stones:
            return 2
        else:
            return 0  # 引き分け

    def get_next_state_from_action(self, action: Dict, player: int) -> np.ndarray:
        """
        指定したアクションを適用した後の盤面を返す（実際には手を打たない）

        引数:
            - action: {"row": int, "col": int} または None（パスの場合）
            - player: プレイヤーID (1または2)

        戻り値:
            - 適用後の盤面のコピー
        """
        # 現在の盤面をコピー
        new_board = self.board.copy()

        # パスの場合
        if action is None:
            return new_board

        row, col = action["row"], action["col"]

        # 無効な手の場合は現在の盤面を返す
        if not self.is_valid_move(row, col, player):
            return new_board

        # 一時的な環境を作成して手を適用
        temp_env = ReversiEnv()
        temp_env.board = new_board
        temp_env.current_player = player

        # 手を適用
        temp_env.make_move(row, col)

        return temp_env.board
