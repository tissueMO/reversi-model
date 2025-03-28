import os
import argparse
import tensorflowjs as tfjs
import tensorflow as tf

def export_to_tfjs(model_path, output_path):
    """
    学習済みのTensorFlowモデルをTensorFlow.js形式にエクスポートする

    Args:
        model_path: TensorFlowモデルのパス
        output_path: 出力ディレクトリのパス
    """
    # モデルを読み込む
    print(f"モデルを読み込み中: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # モデルの構造を表示
    model.summary()

    # 出力ディレクトリを作成
    os.makedirs(output_path, exist_ok=True)

    # TensorFlow.js形式にエクスポート
    print(f"モデルをエクスポート中: {output_path}...")
    tfjs.converters.save_keras_model(model, output_path)

    print(f"モデルのエクスポートに成功しました: {output_path}")

    # モデルのメタデータを作成
    metadata = {
        "format": "layers-model",
        "generatedBy": "TensorFlow.js Converter",
        "convertedBy": "tensorflowjs_converter",
        "modelTopology": {
            "input_shape": model.input_shape[1:],
            "output_shape": [o.shape[1:] for o in model.outputs]
        }
    }

    # メタデータをファイルに保存
    print("メタデータを作成中...")
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    print("エクスポートが完了しました！")


def main():
    parser = argparse.ArgumentParser(description='TensorFlowモデルをTensorFlow.js形式にエクスポートします')
    parser.add_argument('--model-path', type=str, required=True, help='TensorFlowモデルのパス')
    parser.add_argument('--output-path', type=str, default='./export', help='TensorFlow.jsモデルの出力ディレクトリ')

    args = parser.parse_args()

    # TensorFlow.js形式にエクスポート
    export_to_tfjs(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
