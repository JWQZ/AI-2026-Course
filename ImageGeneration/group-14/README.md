# 图像生成 · 第14组：基于视觉条件扩散模型的人脸匿名化

本项目在改变人脸身份的同时，尽量保留原图表情、姿态和自然视觉质量。我们采用 **64候选教师筛选 → 匿名伪监督配对 → CLIP + IP-Adapter Plus + UNet LoRA 微调**，并比较直接训练与两阶段训练路线。最终学生从随机噪声生成整张匿名图，输入为冻结视觉编码器的隐特征，不输入源图像素/latent，不需要用户文本。

## 目录

```text
ImageGeneration/group-14/
  README.md           # 项目介绍、使用入口、结果与局限
  .gitignore          # 仅本组的忽略规则
  code/
    README.md         # 完整安装、权重下载、数据结构和A/B复现命令
    requirements.txt
    THIRD_PARTY.md
    src/              # 特征缓存、模型、训练、推理和checkpoint评估
    teacher/          # 一次教师采样，可多任务并行64轮
    data_selection/   # 固定评分和逐源图64选一
    evaluation/       # ArcFace/姿态/HSE/VA/AU/MUSIQ
    scripts/          # 环境变量控制的运行入口与submission打包
  results/            # 可直接查看的真实实验曲线与固定样例
```

不包含数据集、初始/训练权重、虚拟环境、缓存或提交CSV。课程FFHQ数据仅用于课程，不在公开仓库转发完整数据集。示例图片来自课程数据及对应模型输出，不应另作身份识别用途。

## 环境和使用方法

Python 3.10，匹配CUDA的PyTorch/torchvision；训练使用单节点4张A100-80G、bf16、DDP，支持调整为其他GPU数量。24GB是作业建议最低资源，不是本项目实际实验设备。无需强制安装xformers/flash-attn，使用PyTorch SDPA。完整AU评估还需编译OpenFace。

```bash
cd ImageGeneration/group-14/code
# 先安装与当前CUDA匹配的torch和torchvision，再安装其余依赖
pip install -r requirements.txt
pip install --no-deps pyiqa==0.1.13
export PROJECT_ROOT=/absolute/path/to/workspace
export MODEL_ROOT=/absolute/path/to/models
```

pyiqa仅用于MUSIQ，其其他架构的旧Transformers依赖与本工程冲突，故MUSIQ所需依赖显式列出后单独安装pyiqa。不要用`pip install pyiqa`覆盖当前Transformers版本。

按[代码说明](code/README.md)放置课程数据、下载RV5.1、IP-Adapter Plus/CLIP和官方MSE VAE等权重。下载链接、完整文件结构、OpenFace配置、特征准备与全部脚本参数均在该文档，不依赖私人服务器绝对路径。

复现顺序：

1. 使用正式train/val/test划分（16000/2000/2000），不重新随机拆分。可用 `scripts/sample_teacher.sh` 对训练图采样64轮，`scripts/select_teacher.sh` 逐图按综合分选一张匿名监督图，形成JSONL配对；已有配对时可直接从缓存开始。
2. `SPLIT=train bash scripts/prepare.sh` 缓存 `[257,1280]` CLIP hidden states；训练入口默认也会检查/准备缓存，缺失时才提取。
3. 选择以下A/B训练路线，结束后默认评估完整val上的各checkpoint。

```bash
# 路线A：公开预训练底座初始化，直接使用MSE VAE训练
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=20261031 bash scripts/train_direct.sh

# 路线B：原VAE第一阶段，然后MSE VAE第二阶段
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=1324400183 bash scripts/train_stage1.sh
INIT_CHECKPOINT=/absolute/path/to/stage1_run/epoch_009 \
GPUS=4 BATCH_SIZE=16 GLOBAL_BATCH=64 SEED=20261031 bash scripts/train_stage2.sh
```

A历史最佳epoch7；B第二阶段历史最佳epoch4。B只初始化模型权重，重置优化器/学习率计划/步数，不等于原状态resume。新实验应仍由val选模，不能根据test调参。

```bash
# 对test单次生成并打包，不按test分数采样择优
SPLIT=test bash scripts/prepare.sh
CHECKPOINT=/absolute/path/to/best_checkpoint \
CONDITIONS="$PROJECT_ROOT/data/features_v4/test/conditions.jsonl" \
OUTPUT="$PROJECT_ROOT/outputs/test_generation" SEED=20261026 bash scripts/infer.sh
GENERATED="$PROJECT_ROOT/outputs/test_generation" \
SUBMISSION="$PROJECT_ROOT/outputs/submission.csv" bash scripts/submission.sh
```

## 方法与训练细节

冻结CLIP ViT-H/14提取原图倒数第二层token `[B,257,1280]`，预训练IP-Adapter Plus Resampler映射到`[B,16,768]`。图像token经独立K/V注意力注入Realistic Vision V5.1四通道UNet。训练Resampler、IP图像K/V及UNet attention LoRA；冻结CLIP、VAE和UNet原权重，文本分支使用固定空条件。

监督是64选一匿名目标y，而非原图x。VAE编码目标并按DDPM时间步加噪，训练标准去噪MSE：

$$z_t=\sqrt{\bar\alpha_t}E(y)+\sqrt{1-\bar\alpha_t}\epsilon,\qquad L=\mathbb E\|\epsilon_\theta(z_t,t,C(x))-\epsilon\|^2/(CHW).$$

没有额外像素/身份/表情/LPIPS训练损失；这些属性主要通过伪监督筛选保持。10%原始CLIP条件置零，经同一Resampler学习空条件；推理CFG为`epsilon_null + 3*(epsilon_cond-epsilon_null)`。IP_SCALE=.8是图像注意力强度，不是CFG。

每阶段10epochs、2500更新，global batch64；AdamW学习率1e-4、weight decay.01、cosine schedule、warmup.05，LoRA rank/alpha32、dropout.05，512px。最终推理官方MSE VAE、DPM++ 2M Karras、30步、CFG3、IP.8。B最佳继承第一阶段2250更新再新增1000更新，A最佳1750更新，不能声称相同总算力下B更好。

## 结果展示与评估口径

所有单项按主脸评估：ArcFace身份cos越低越不像原身份；姿态角MAE越低越好；HSE隐特征cos越高越好；VA/AU误差越低越好；MUSIQ越高代表较高感知质量，但不等于照片真实性。评测256px，生成图JPEG75；AU由OpenFace静态回归，仅在有效配对上统计。单项均值分母可能不同。项目自定义Q逐图计算，必要配对失败记0、分母固定2000，不是Kaggle官方评分，不用各项均值拼造Q。

Q权重为身份.25、姿态.20、HSE.20、VA.10、MUSIQ.20、有效性.05；各项归一化依次为clip((.35-id)/.35)、exp(-pose/15)、clip((hse-.2)/.8)、exp(-VA/.5)、clip((musiq-20)/60)，clip到[0,1]。AU不参与Q。

| 模型 | 划分/规模 | Q↑ | 身份cos↓ | 姿态°↓ | HSEcos↑ | VA↓ | AU↓ | MUSIQ↑ | 身份有效 | AU有效 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Baseline一次采样 | test/2000，跨划分参考 | .68860 | .05096 | 4.50925 | .49138 | .25588 | .49033 | 63.08109 | 1999/2000 | 1757/2000 |
| A最佳epoch7 | val/2000 | .73782 | .05415 | 3.18187 | .57784 | .16069 | .39901 | 65.21270 | 2000/2000 | 1815/2000 |
| B最佳新epoch4 | 同一val/2000 | .74460 | .05138 | 3.11604 | .58235 | .15771 | .39354 | 65.86755 | 2000/2000 | 1801/2000 |

Baseline与A/B不在同一划分，不能据此宣称严格测试提升。A/B使用相同val推理配置；B仅有约.00678的Q优势，没有多seed、等总预算或显著性检验。两条最佳checkpoint均另已完成test2000单次推理和CSV逐条校验，但**未重算完整test指标、未报告Kaggle返回分数**。提交CSV另行交付，不上传Git。

### 同源验证前五张：原图、A最佳、B最佳

![同源验证样例](results/final_AB_five.png)

### 同源test前五张：原图、Baseline、A、B

![test四路对比](results/test_final_four_way.png)

固定前五张，未按效果挑选；验证集没有教师64轮采样，不把训练监督图冒充验证集同源监督。

### 采样累计最优与教师监督

固定8张训练源图×64候选=512记录，逐源图、逐指标独立取历史best后求均值；Q在真实候选Q中独立取best，不拼接不同指标最优。K表示实际候选数，身份/姿态/VA取min，HSE/MUSIQ/Q取max。Q(K=1/8/32/64)=.692319/.782595/.807468/.821399，后段收益趋缓。这是小样本诊断，不代表全部训练集。

![采样收益](results/sampling64.png)
![原图、首次采样、64最佳](results/teacher_best64.png)

### A/B各epoch验证曲线

![A路线](results/routeA.png)
![B路线](results/routeB.png)

## 局限与展望

仍存在皮肤过度平滑、妆感、外观年轻化、局部表情和眼镜等配饰变化；MUSIQ和身份cos不能替代人工真实性检查，也不能证明绝对匿名。可能由教师美颜先验、缺少真实纹理筛选项和监督偏差导致；这是合理解释，尚未独立验证。

后续可固定多样人像小样本对比FLUX Fill/SDXL等teacher，加入自然皮肤纹理和人工盲评，再决定是否重构监督；更换teacher不要求直接替换学生架构。还需要相同test上的baseline/A/B定量对比、多seed等预算实验、独立身份识别器及失败样本分组统计。以上均为未来工作，不作为已完成结果。

模型和方法来源见[第三方说明](code/THIRD_PARTY.md)及[权重下载说明](code/README.md)。完整数据和模型权重按各自上游许可单独获取。
