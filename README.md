# リバーシAI 強化学習モデル

このプロジェクトは、強化学習を用いてリバーシ（オセロ）のAIモデルを学習し、ブラウザ上で実行可能なTensorFlow.js形式でエクスポートするものです。

## プロジェクト概要

- Python/TensorFlowによる強化学習モデルの学習
- 学習したモデルをTensorFlow.js形式にエクスポートしブラウザ上で推論可能
- 自己対戦による効率的な学習プロセス
- 軽量なモデル構造によるブラウザでの高速推論
- TensorBoardによる学習過程と評価結果の可視化

## 環境要件

- Python 3.8以上
- TensorFlow 2.x
- TensorFlow.js
- NumPy
- matplotlib
- tqdm

## プロジェクト構成

```
/app/
├── src/                # ソースコード
│   ├── game/           # リバーシゲームロジック
│   │   ├── __init__.py
│   │   └── reversi.py  # リバーシのルールとゲーム管理
│   ├── model/          # AIモデル定義
│   │   ├── __init__.py
│   │   └── model.py    # ニューラルネットワークモデルの構造
│   ├── train.py        # モデル学習スクリプト
│   └── export_tfjs.py  # TensorFlow.js形式へのエクスポート
├── tests/              # テストコード
│   ├── __init__.py
│   └── test_reversi.py # リバーシロジックのテスト
├── output/             # 出力ファイル
│   ├── models/         # 保存されたモデル
│   ├── logs/           # TensorBoardログ
│   └── export/         # エクスポートされたTensorFlow.jsモデル
├── requirements.txt    # 依存ライブラリ
├── Dockerfile          # Dockerビルド設定
└── docker-compose.yml  # Docker Compose設定
```

## インストール方法

依存ライブラリをインストールします：

```bash
pip install -r requirements.txt
```

## 使い方

### モデルの学習

```bash
python src/train.py --games 100 --iterations 10 --epochs 10 --batch-size 128 --temperature 1.0 --output-dir ./output/models --evaluate --visualize
```

主なオプション：
- `--games`: 各イテレーションで行う自己対戦ゲーム数
- `--iterations`: 学習のイテレーション数
- `--epochs`: 各イテレーションでの学習エポック数
- `--batch-size`: バッチサイズ
- `--temperature`: 探索の温度パラメータ（高いほど多様な手を試す）
- `--output-dir`: モデルと学習グラフの保存先
- `--log-dir`: TensorBoardログの保存先（デフォルト: ./logs）
- `--evaluate`: トレーニング後にモデル強度を評価するフラグ
- `--visualize`: 評価中のゲーム状態を可視化するフラグ

### TensorBoardによる学習状況の可視化

学習中または学習後に以下のコマンドでTensorBoardを起動できます：

```bash
tensorboard --logdir=./logs
```

ブラウザで http://localhost:6006 にアクセスして以下の情報を確認できます：

1. **SCALARSタブ**：
   - 学習損失（ポリシー損失、価値損失、合計損失）の推移
   - モデル評価の結果（勝率、引き分け率、敗北率）

2. **IMAGESタブ**：
   - リバーシ盤面の可視化（`--visualize`オプション使用時）
   - 対局の最終状態

3. **TEXTタブ**：
   - 各評価ゲームの結果（勝敗、スコア）

4. **HISTOGRAMSタブ**：
   - モデルの重みとバイアスの分布

学習進捗をリアルタイムで監視でき、モデルの改善に役立ちます。

### モデルのエクスポート

```bash
python src/export_tfjs.py --model-path ./output/models/<training_timestamp>/final_model/reversi_model --output-path ./output/export
```

オプション：
- `--model-path`: エクスポートする学習済みモデルのパス
- `--output-path`: エクスポート先ディレクトリ

## モデル仕様

### 入力仕様
現在の盤面を8x8の二次元配列として入力します：
- 0: 空きマス
- 1: プレイヤー1の石
- 2: プレイヤー2の石

内部では、これを3チャンネルの形式に変換します：
- チャンネル0: 自分の石 (1または0)
- チャンネル1: 相手の石 (1または0)
- チャンネル2: 空きマス (1または0)

### 出力仕様
モデルの出力は以下の2つです：
1. **次に置くべき座標**: `{ row: number, col: number }` 形式、有効な手がない場合は `null`
2. **予想勝率**: 0〜1の値で現在の盤面からの勝率を示す

### モデル構造
- 入力層: 8x8x3 (盤面の3チャンネル表現)
- 畳み込み層: 複数の畳み込み層とバッチ正規化
- 政策ヘッド: 次の一手の確率分布（64マス+パス=65次元）
- 価値ヘッド: 現在の盤面からの勝率予測（-1〜1）

## モデル評価と強さの可視化

学習済みモデルを評価して強さを確認できます：

```bash
python train.py --evaluate --visualize --log-dir=./output/logs/evaluation
```

この評価機能では以下が行われます：
- ランダムプレイヤーとの対戦による勝率の測定
- モデル間の対戦による相対的な強さの比較
- TensorBoardでの対局結果と盤面状態の可視化

評価結果は以下の指標で確認できます：
- 勝率 (%)
- 引き分け率 (%)
- 平均獲得石数

## フロントエンドでの使用例

TensorFlow.jsを使用してブラウザ上でモデルを読み込み、推論を行う例：

```javascript
// モデルの読み込み
const model = await tf.loadLayersModel('path/to/model.json');

// 盤面の前処理
function preprocessBoard(board, player) {
  // 盤面を3チャンネル形式に変換
  const input = tf.tensor4d(/* 変換処理 */, [1, 8, 8, 3]);
  return input;
}

// 推論実行
async function predictMove(board, player, validMoves) {
  const input = preprocessBoard(board, player);
  const [policyPred, valuePred] = await model.predict(input);
  
  // 政策から最適な手を選択
  const policy = await policyPred.data();
  let bestMove = findBestMove(policy, validMoves);
  
  // 勝率の取得
  const winRate = (await valuePred.data()[0] + 1) / 2; // -1〜1 を 0〜1 に変換
  
  return { move: bestMove, winRate: winRate };
}
```

## ライセンス

MITライセンス
