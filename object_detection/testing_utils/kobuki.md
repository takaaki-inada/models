


## 物体自動追尾

元のコードは ssd_keras のサンプルプログラムで、WEBCAM の画像から物体を検出し、検出した物体ごとの確度をOpenCVで表示するもの

これに、簡単な追尾用に以下の改造をほどこしたもの
- 指定した物体の検出のみに特定
- 前回検出した位置と一番近い位置を取得
- この位置に対して物体がCENTERから左にあれば右に、右にあれば左に、CENTER内にあれば前進するようros連携

## Getting Start

This code was tested with `Tensorflow` v1.2.0, `OpenCV` v3.2.0

### 準備

- このブランチを取得
```
git clone -b feature-ros-kobuki https://github.com/takaaki-inada/models.git
```
- object_detection/g3doc/installation.md object_detection/object_detection_tutorial.ipynb の記載の通りセットアップを済ませる
- OpenCVのセットアップ(以下リンク等を参考)  
https://milq.github.io/install-opencv-ubuntu-debian/
- WEBCAM を /dev/video0 に認識させる
- 追尾する物体をコード上に設定 (videotest_example.py の最後のほうの行)  
conf_thresh: 確度がこの閾値以下の場合は検出した物体を無視  
target_class_label: 追尾する物体のクラス(実行時に最初に表示しているcategory_indexを参照)
```
vid_test.run(conf_thresh = 0.5, target_class_label = 1)
```

### 実行方法

```
cd ${project_root}
python object_detection/testing_utils/videotest_example.py
```
