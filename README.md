# Neural Style Transfer Project

这是一个基于 VGG19 的 Neural Style Transfer 项目，用于将风格图像的纹理、色彩和笔触迁移到内容图像上，同时尽量保留原始内容结构。

项目实现围绕内容损失、风格损失和总变差约束展开，完整记录了从早期效果不稳定到最终成功风格化的实验过程。

## 项目目标

- 使用 VGG19 提取内容特征和风格特征
- 基于 Gram Matrix 计算风格损失
- 通过优化生成图像完成风格迁移
- 观察不同损失权重对结果的影响
- 保存训练过程中的中间结果图

## 目录结构

```text
style_transfer_project/
├── assets/
│   ├── content/content.jpg   # 内容图像
│   └── style/style.jpg       # 风格图像
├── models/
│   └── style_transfer.py     # VGG 特征提取与 Gram Matrix
├── outputs/                  # 生成结果保存目录
├── utils/
│   └── image_utils.py        # 图像加载与保存
├── train.py                  # 风格迁移主入口
├── requirements.txt          # 依赖列表
└── README.md
```

## 核心方法

### 内容损失

使用 `conv4_2` 层特征作为内容表示，约束生成图像保持原图结构。

### 风格损失

从多个卷积层提取风格特征，并通过 Gram Matrix 衡量通道之间的相关性，从而捕捉风格纹理。

### 总变差损失

用于抑制生成图像中的高频噪声，使结果更平滑、更稳定。

## 模型说明

`models/style_transfer.py` 中的 `VGGFeatures` 使用预训练 VGG19 的部分卷积层作为特征提取器：

- `conv1_1`
- `conv2_1`
- `conv3_1`
- `conv4_1`
- `conv4_2`
- `conv5_1`

其中：

- `conv4_2` 用于内容约束
- 其余层用于风格约束

## 运行方式

先安装依赖：

```bash
pip install -r requirements.txt
```

然后执行：

```bash
python train.py
```

脚本会读取：

- `assets/content/content.jpg`
- `assets/style/style.jpg`

并将结果保存到 `outputs/` 目录。

## 训练设置

当前主流程中采用的参数如下：

- 优化器：`Adam`
- 学习率：`0.02`
- 迭代步数：`800`
- 内容权重：`1e2`
- 风格权重：`1e7`
- TV 权重：`0.0`

这些参数决定了最终图像在“保留内容”和“迁移风格”之间的平衡。

## 实验过程

项目记录了几个典型阶段：

1. 早期版本中，风格迁移不明显
2. 加入 TV loss 后，噪声有所降低，但风格仍偏弱
3. 调整内容权重与风格权重后，生成图像开始出现稳定的风格化效果

从结果上看，背景区域通常会先被风格化，主体人物或主体物体则在后期逐步被风格覆盖，这与内容约束和风格约束的平衡有关。

## 输出结果

训练过程中会定期保存中间图像，例如：

- `outputs/step_0.jpg`
- `outputs/step_100.jpg`
- `outputs/step_200.jpg`
- `outputs/step_300.jpg`
- `outputs/step_400.jpg`
- `outputs/step_500.jpg`
- `outputs/step_600.jpg`
- `outputs/step_700.jpg`

最终结果保存为：

- `outputs/stylized_version2.jpg`

## 结果分析

这个项目的结论比较明确：

- 内容权重过高时，风格不容易显现
- 风格权重过高时，图像容易失去内容结构
- TV loss 有助于减少噪声，但不是风格迁移的核心驱动项
- 生成过程本质上是对图像像素本身进行优化，因此收敛速度和效果都高度依赖权重设置

## 备注

- `torchvision` 会加载预训练 VGG19 权重
- 首次运行时需要确保网络环境可下载预训练模型
- `outputs/` 中已经保存了多个阶段性结果，便于对比观察

