import numpy as np
import tensorflow as tf


class VisualizationManager:
    """ゲーム盤面の可視化を管理するユーティリティクラス"""

    @staticmethod
    def visualize_board(board, move_history=None):
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
                    VisualizationManager._draw_circle(vis_board, r_center,
                                    c_center, radius, [0, 0, 0])
                elif board[row, col] == 2:  # プレイヤー2の石（白）
                    VisualizationManager._draw_circle(vis_board, r_center,
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
                        VisualizationManager._draw_number(vis_board, r_center,
                                        c_center, move_number, color, size)

        # バッチ次元を追加（TensorBoardのimage summaryのため）
        return np.expand_dims(vis_board, axis=0)

    @staticmethod
    def _draw_circle(image, center_r, center_c, radius, color):
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

    @staticmethod
    def _draw_number(image, center_r, center_c, number, color, size):
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
            VisualizationManager._draw_digit(image, digit, center_r,
                             digit_center_c, color, size)

    @staticmethod
    def _draw_digit(image, digit, center_r, center_c, color, size):
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
        segment_points = VisualizationManager._get_segment_points(
            center_r, center_c, half_height, half_width)

        # 対応するセグメントを描画
        segment_patterns = segments.get(digit, segments['0'])  # 未定義の文字は0として扱う

        for i, active in enumerate(segment_patterns):
            if active:
                # セグメントの描画（セグメントごとに異なる描画処理）
                start_point, end_point = segment_points[i]
                if i in [0, 3, 6]:  # 水平なセグメント（上、下、中央）
                    VisualizationManager._draw_horizontal_segment(
                        image, start_point, end_point, color, thickness)
                else:  # 垂直なセグメント
                    VisualizationManager._draw_vertical_segment(
                        image, start_point, end_point, color, thickness)

    @staticmethod
    def _get_segment_points(center_r, center_c, half_height, half_width):
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

    @staticmethod
    def _draw_horizontal_segment(image, start_point, end_point, color, thickness):
        """水平なセグメントを描画"""
        r, c_start = start_point
        _, c_end = end_point

        for c in range(c_start, c_end + 1):
            for t in range(-thickness, thickness + 1):  # 太さを増加
                if 0 <= r + t < image.shape[0] and 0 <= c < image.shape[1]:
                    image[r + t, c] = color

    @staticmethod
    def _draw_vertical_segment(image, start_point, end_point, color, thickness):
        """垂直なセグメントを描画"""
        r_start, c = start_point
        r_end, _ = end_point

        for r in range(r_start, r_end + 1):
            for t in range(-thickness, thickness + 1):  # 太さを増加
                if 0 <= r < image.shape[0] and 0 <= c + t < image.shape[1]:
                    image[r, c + t] = color
