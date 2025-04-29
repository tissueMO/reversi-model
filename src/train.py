import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import psutil
from datetime import datetime

from src.game import ReversiEnv
from src.model import ReversiModel
from src.trainer import ReversiTrainer


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
                        default='./output/models', help='モデルの保存先ディレクトリ')
    parser.add_argument('--log-dir', type=str,
                        default='./output/logs', help='TensorBoardログの保存先ディレクトリ')
    parser.add_argument('--evaluate', action='store_true',
                        help='学習後にモデルの強さを評価する')
    parser.add_argument('--visualize', action='store_true',
                        help='評価中のボード状態を可視化する')
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
        trainer = ReversiTrainer.load_from_model(
            model_path=args.load_model,
            buffer_size=args.buffer_size,
            log_dir=args.log_dir
        )
        start_iteration = args.start_iteration
    else:
        trainer = ReversiTrainer(
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

        # 中間モデルを保存
        trainer.save_model(path=os.path.join(
            output_dir, f"model_iter_{i+1}"))

    # 最終モデルを保存
    trainer.save_model(path=os.path.join(output_dir, "final_model"))
    trainer.remove_interim_models(output_dir)

    # モデルの評価
    if args.evaluate:
        print("\nモデルの強さを評価中...")
        evaluation_results = trainer.evaluate_model_strength(
            num_games=args.games, visualize=args.visualize)
        print(f"評価結果: 勝ち: {evaluation_results['wins']}回, 引き分け: {evaluation_results['draws']}回, " +
              f"負け: {evaluation_results['losses']}回, 勝率: {evaluation_results['win_rate']:.2%}")

    print(f"\n学習が完了しました。モデルが {output_dir}/final_model に保存されました")
    print("すべての中間モデルは削除されました")
    print(f"TensorBoardログは {trainer.log_dir} に保存されました")
    print(f"学習の進捗をTensorBoardで確認するには: tensorboard --logdir={args.log_dir}")
    print("その後、ブラウザで http://localhost:6006 を開いてください")


if __name__ == "__main__":
    main()
