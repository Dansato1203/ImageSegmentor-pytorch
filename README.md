# soccerfield_detector  
  
## Usage
1. データセット（オリジナル画像とラベル画像がセットになったもの）をドライブからダウンロードする。
  
2. zipであれば解凍し、soccerfield_detectorディレクトリの下に配置する。  
```
soccerfield_detector/  
　├ データセット/  
　│　　　├ *.jpg
　│　　　└ *.png
　├ scripts/  
　├ Docker/  
　└ src/  
 ```  
  
 3. Dockerfileのデータセットの名前を変更する  
https://github.com/Dansato1203/soccerfield_detector/blob/9dd641a61ec6c574ef3f4910f9d782eda6df1dfe/Docker/Dockerfile#L25
変更するのは、**上記の前の部分（220623_dataset/train_dataset）** のみ  
中にjpgファイルとpngファイルが入っているディレクトリの名前に変更する   
  
4. Docker build  
```bash
# soccerfield_detectorディレクトリ上で
Docker build -t pytorch_train:{タグ} -f Docker/Dockerfile .
```  
  
5. 学習開始  
上記でつけたタグ名にlaunch_train.sh内のタグを変更する  
https://github.com/Dansato1203/soccerfield_detector/blob/9dd641a61ec6c574ef3f4910f9d782eda6df1dfe/launch_train.sh#L2
変更したのちスクリプトを実行する  
```
./launch_train.sh
```  
  
6. 重みのアップロード
学習が終わるとtrain_resultディレクトリが生成され、その中に推論した画像と重みが保存される  
これをzip等に圧縮した後、[ドライブ](https://drive.google.com/drive/folders/1wbTg-Qi8e7yRFnhCzvoHUJVgPb4OyEwH?usp=sharing)にアップロードする
