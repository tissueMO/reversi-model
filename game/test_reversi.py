import unittest
import numpy as np
from game.reversi import ReversiEnv

class TestReversiEnv(unittest.TestCase):
    def setUp(self):
        """各テストの前に実行される前処理"""
        self.env = ReversiEnv()

    def test_初期盤面(self):
        board = self.env.board
        self.assertEqual(board[3, 3], 1)  # 白石
        self.assertEqual(board[4, 4], 1)  # 白石
        self.assertEqual(board[3, 4], 2)  # 黒石
        self.assertEqual(board[4, 3], 2)  # 黒石

        # 他のマスが空であることを確認
        self.assertEqual(np.sum(board), 6)  # 初期配置の石の合計値は6 (1*2 + 2*2)
        self.assertEqual(np.sum(board == 0), 60)  # 60マスが空

    def test_初期状態での有効な手(self):
        valid_moves = self.env.get_valid_moves(1)  # プレイヤー1の有効な手
        expected_moves = [(2, 4), (3, 5), (4, 2), (5, 3)]  # 実際の実装での正しい手

        self.assertEqual(len(valid_moves), len(expected_moves))
        for move in expected_moves:
            self.assertIn(move, valid_moves)

    def test_石を置いたときのひっくり返し(self):
        # プレイヤー1が(2,4)に石を置く
        self.env.make_move(2, 4)

        # (2,4)に石が置かれ、(3,4)がひっくり返されたことを確認
        self.assertEqual(self.env.board[2, 4], 1)
        self.assertEqual(self.env.board[3, 4], 1)

        # プレイヤーが切り替わっていることを確認
        self.assertEqual(self.env.current_player, 2)

    def test_有効な手がない場合の自動パス(self):
        # カスタムの盤面を設定：プレイヤー2に有効な手がない状況
        custom_board = np.zeros((8, 8), dtype=np.int8)
        custom_board[3, 3:5] = 1  # プレイヤー1の石
        custom_board[4, 3:5] = 1  # プレイヤー1の石

        self.env.board = custom_board
        self.env.current_player = 2

        # プレイヤー2に有効な手がないことを確認
        valid_moves = self.env.get_valid_moves(2)
        self.assertEqual(len(valid_moves), 0)

        # 適当な値で手を実行すると自動的にパスされる
        _, _, _, _ = self.env.make_move(0, 0)
        self.assertEqual(self.env.current_player, 1)  # プレイヤー1に切り替わる

    def test_ゲーム終了と勝者判定(self):
        # ゲーム終了状態の盤面を設定
        custom_board = np.ones((8, 8), dtype=np.int8)
        custom_board[0:4, :] = 1  # 上半分はプレイヤー1
        custom_board[4:8, :] = 2  # 下半分はプレイヤー2

        self.env.board = custom_board
        self.env.current_player = 1

        # 両プレイヤーとも手がないのでゲームは終了する
        # まず、有効な手がないことを確認
        valid_moves1 = self.env.get_valid_moves(1)
        self.assertEqual(len(valid_moves1), 0)
        valid_moves2 = self.env.get_valid_moves(2)
        self.assertEqual(len(valid_moves2), 0)

        # make_moveを呼び出すとゲーム終了フラグがセットされる
        _, _, done, _ = self.env.make_move(0, 0)

        # ゲームが終了しているか確認
        self.assertTrue(done)
        self.assertTrue(self.env.done)

        # 盤面の状態から勝者を確認
        self.assertEqual(self.env.get_winner(), 0)  # 引き分け

    def test_複雑なひっくり返しパターン(self):
        # カスタムの盤面を設定
        custom_board = np.zeros((8, 8), dtype=np.int8)
        # 縦、横、斜めに石を配置
        custom_board[2, 2] = 1
        custom_board[2, 3] = 2
        custom_board[2, 4] = 2
        custom_board[3, 2] = 2
        custom_board[3, 3] = 2
        custom_board[3, 4] = 2
        custom_board[4, 2] = 2
        custom_board[4, 3] = 2
        custom_board[4, 4] = 1

        self.env.board = custom_board
        self.env.current_player = 1

        # (2,5)に石を置くと(2,3)と(2,4)がひっくり返る
        self.assertTrue(self.env.is_valid_move(2, 5, 1))
        self.env.make_move(2, 5)

        self.assertEqual(self.env.board[2, 3], 1)
        self.assertEqual(self.env.board[2, 4], 1)
        self.assertEqual(self.env.board[2, 5], 1)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
