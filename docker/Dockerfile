FROM nvcr.io/nvidia/pytorch:22.12-py3

# 言語の設定
RUN apt update && apt install -y sudo language-pack-ja-base language-pack-ja
ENV LANG='ja_JP.UTF-8'

# 必要なもののインストール
RUN apt install -y vim
RUN pip3 install timm

# WORKDIRを設定
WORKDIR /root

# dotfileの導入
RUN git clone https://github.com/SakodaShintaro/dotfiles && ./dotfiles/setup.sh
