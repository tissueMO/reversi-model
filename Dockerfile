FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーしてパッケージをインストール
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 作業ディレクトリをマウント予定のため、ここではファイルをコピーしない
# コンテナ起動時にコマンドを実行しない（docker-composeで指定）

# TensorBoardのポートを公開
EXPOSE 6006
