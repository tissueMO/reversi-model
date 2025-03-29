FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y git curl python3-pip python3-dev python3-opencv libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6006

CMD ["tensorboard", "--logdir=logs", "--host=0.0.0.0", "--port=6006"]
