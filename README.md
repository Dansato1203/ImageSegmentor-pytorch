# ImageSegmentor-pytorch

## 概要
ImageSegmentor-pytorchは，3クラスの領域を対象とした汎用的なセマンティックセグメンテーションツールです．  
このリポジトリには，PythonによるテストコードおよびLibTorch (C++) によるテストコードが含まれています．  

## 特徴
- 3クラスのオブジェクトに対応する柔軟なセマンティックセグメンテーション機能
- PythonとLibTorch (C++) 両方でのテストコード提供
- カスタマイズが容易な学習パイプライン

## requirements
ImageSegmentor-pytorchを使用するためには，以下のライブラリが必要です．
- PyTorch >= 1.9
- numpy
- OpenCV
- Matplotlib

  
## 使用法
1. データセットを準備する
   
1.1. サンプルデータセットは以下にあります． （サッカーフィールドの白線，芝生のデータセット）
  

4. With Docker   
```bash
Docker build -t pytorch_train:{IMAGE_NAME} -f Docker/Dockerfile .
```  
  
5. 学習  
以下のスクリプトで学習を開始できます．  
```
./launch_train.sh ${IMAGE_NAME}
```  
  
6. 重みのアップロード
学習が終わるとtrain_resultディレクトリが生成され、その中に推論した画像と重みが保存される  
これをzip等に圧縮した後、[ドライブ](https://drive.google.com/drive/folders/1wbTg-Qi8e7yRFnhCzvoHUJVgPb4OyEwH?usp=sharing)にアップロードする
