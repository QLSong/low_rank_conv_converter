# low rank conv converter
- 这是一个给pytorch模型用的工具，可以给pytorch模型的卷积操作添加lora和DnA（目前只支持卷积层的替换）
- 可用于少样本的训练，以及大模型的快速迭代

## 相关链接
- DnA: Improving Few-shot Transfer Learning with Low-Rank Decomposition and Alignment
  - https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136800229.pdf
  - https://github.com/VITA-Group/DnA.git
- LoRA: Low-Rank Adaptation of Large Language Models
  - https://arxiv.org/pdf/2106.09685v2.pdf
  - https://github.com/microsoft/LoRA

# 安装
Install commands:
```shell
git clone https://github.com/QLSong/low_rank_conv_converter.git
cd low_rank_conv_converter
python setup.py install
```

# 用法
```python
from torchvision.models.resnet import resnet18, ResNet18_Weights
import lrcc
# model = resnet18(pretrained=True)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = lrcc.model_convert_conv(model, mode='lora_mode', ignore_op_name=['downsample'])
```
## 参数说明
```python
model_convert_conv(
    model, # pytorch模型
    mode, # 微调方法选择，['fix_sparse', 'fix_low_rank', 'tune_U', 'tune_V', 'tune_V_S', 'tune_U_S', 'tune_all', 'lora_mode']
    path=None, # pretrain模型路径，lora_mode可以设为None，其他需要有值
    convert_op_name=[], # 需要转化的层，例如convert_op_name=['conv']，就会把所有名字带conv的op转化成low_rank_conv
    ignore_op_name=[], # 忽视转化的层，例如ignore_op_name=['downsample']，就会把所有名字不带downsample的op转化为low_rank_conv，注意convert_op_name和ignore_op_name只能设置一个
    r=0, # 分解后UV的隐藏维度
    r_ratio=0.25, # 分解后UV的隐藏维度=input_channels*r_ratio，r_ratio与r只需要设置一个
    compress_step=50, # 分解UV是QR分解的迭代次数
    lambda_s=0.01 # UV分解后，weight-UV>lambda_s的部分作为sparse_mask
)
```
# 测试
在分类模型resnet18上做了简单测试，pretrain是imagenet，finetune数据集是caltech101，每个类别使用若干张图片训练，其他图片作为测试
<p align="center">
<table>
  <tr>
    <th>method</th>
    <th>10张训练acc(%)</th>
    <th>5张训练acc(%)</th>
  </tr>
  
  <tr>
    <td>整体finetune</td>
    <td>89.16</td>
    <td>80.94</td>
  </tr>
  
  <tr>
    <td>lora_mode, r_ratio=0.25</td>
    <td>89.27</td>
    <td>82.57</td>
  </tr>

  <tr>
    <td>tune_U_S, r_ratio=0.25</td>
    <td>89.32</td>
    <td>81.13</td>
  </tr>

</table>
</p>







