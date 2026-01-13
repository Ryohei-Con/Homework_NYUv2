## 方針
SegFormerの事前学習モデルを使用(NYUv2とのドメイン親和性を考え、ade20k-finetunedを採用)<br>
自己教師あり学習によりドメイン適応能力を向上<br>
画像拡張をする<br>
ラベル分布の不均衡に対応するために、損失関数の重みを変更する<br>
深度マップに対してHHA encodingし、こちらだけSSLをする<br>
Discriminative Cross-Modal Attention Approach for RGB-D Semantic Segmentationを参考にして階層的にRGBとDepthを融合していく。階層構造はSegFormerに従う<br>
MAEはSegformerのConvと相性が悪いため、断念。代わりにSimMIMを採用し、エッジや画像の端としてみなされることを防ぐため平均値で穴埋め<br>
RGBとHHA-encoded Depthを融合することも考え、比較する<br>

## 注意
depthは0~255の範囲ではないので、vscodeのそのままの表示ではうまく見れない<br>
境界線の前処理<br>
テストデータを用いた自己教師あり学習は禁止<br>
dropout層を増やしたい<br>
<br>
## 引用
@misc{Depth2HHA-python,
    title={{Depth2HHA-python}: Converting depth maps to HHA encodings},
    author={Xiaokang Chen},
    howpublished = {\url{https://github.com/charlesCXK/Depth2HHA-python}},
    year={2018}
}<br>
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}

## 反省
Mask2Formerの方がモダンだった（リサーチ不足）<br>
HHA encodingはよくわかっていない<br>
融合方法をほかにも試して精度比較してみたい。（cross-attentionなど）<br>
optunaを使ったハイパーパラメータ探索をしてみたい（計算コスト大きい？）<br>

## 参考文献
Discriminative Cross-Modal Attention Approach for RGB-D Semantic Segmentation<br>
Learning Rich Features from RGB-D Images for Object Detection and Segmentation<br>
SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers<br>
Segformer公式実装: https://github.com/NVlabs/SegFormer<br>
Masked Autoencoders Are Scalable Vision Learners<br>
SimMIM: A Simple Framework for Masked Image Modeling<br>
