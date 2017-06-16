# imagenet_nogizaka

## 環境
ubuntu14
cuda8.0

    pip install chainer==1.21.0

## データセットをとってくる
https://www.dropbox.com/sh/3i3r68f63nzis9k/AAAeAI1W3FeyXWZWGe2PcRTQa?dl=0
<br>
## 学習

    python make_train_data.py train_image
    python compute_mean.py train.txt
    python train_imagenet.py ./train.txt ./test.txt -m ./mean.npy -g 0 -E 400 -a alex

## 推定

    python test_imagenet.py --test image_list.txt -g 0 -E 1 -m mean.npy --initmodel alex_400e_png_model.h5 -a alex



## 記事
http://qiita.com/miyamotok0105/items/dbf5897013c24f84bd35
<br>
http://qiita.com/miyamotok0105/items/d7cbf32667d831153cd2
<br>