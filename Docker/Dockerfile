FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG weight_name="traced_sf_model.pt"

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y --no-install-recommends \
	vim \
	python3 \
	python3-pip \
	libgl1-mesa-dev \
	libgtk2.0-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip \
	&& pip3 install opencv-python==4.3.0.38 \
	numpy \
	matplotlib \
	scikit-learn \
	torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
	gdown

WORKDIR /train

COPY 220611_dataset/train_dataset /train/train_dataset
COPY 220611_dataset/test_dataset /train/test_dataset
COPY scripts/MLP.py /train/
COPY scripts/train.py /train/
COPY scripts/model_trace.py /train/
COPY Docker/run_train.sh /train/

CMD ["./run_train.sh", "$weight_name"]
