import os
import argparse
import tensorflowjs as tfjs
import tensorflow as tf


def export(model_path, output_path):
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

    # TensorShape を Python のリストに変換する関数
    def shape_to_list(shape):
        if shape is None:
            return None
        return [dim if dim is not None else -1 for dim in shape]

    # モデルのメタデータを作成
    input_shape = shape_to_list(
        model.input_shape[1:]) if model.input_shape else None
    output_shapes = [shape_to_list(o.shape[1:])
                     for o in model.outputs] if model.outputs else []

    metadata = {
        "format": "layers-model",
        "generatedBy": "TensorFlow.js Converter",
        "convertedBy": "tensorflowjs_converter",
        "modelTopology": {
            "input_shape": input_shape,
            "output_shape": output_shapes
        }
    }

    # メタデータをファイルに保存
    print("メタデータを作成中...")
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        import json
        json.dump(metadata, f, indent=2)

    print("エクスポートが完了しました！")


def main():
    parser = argparse.ArgumentParser(
        description='TensorFlowモデルをTensorFlow.js形式にエクスポートします')
    parser.add_argument('--model-path', type=str,
                        required=True, help='TensorFlowモデルのパス')
    parser.add_argument('--output-path', type=str,
                        default='./output/export', help='TensorFlow.jsモデルの出力ディレクトリ')

    args = parser.parse_args()

    # TensorFlow.js形式にエクスポート
    export(args.model_path, args.output_path)


if __name__ == "__main__":
    main()
