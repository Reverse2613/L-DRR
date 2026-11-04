# experiments

我已经从RiO-DETR里面抽取出来的最少最小可训练与测试的源代码，作为我本次自己的实验源代码仓库。

## Install and Getting Start

```
conda create -n rio-detr python=3.10 -y
conda activate rio-detr

# Install PyTorch and torchvision for your CUDA version first.

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# if network is no good, maybe u can try add "-i https://pypi.tuna.tsinghua.edu.cn/simple"
pip install -r requirements.txt

```

## 数据集

### ATR-UMOD

- atr-umod_dataset.py（atr-umod数据集训练时数据加载器代码，未完全懂❌）
- atrumod_condition_parser.py  （显示物理条件解析器，用于推理测试时在不同条件的对比，对单张图片进行六个维度的条件先验分类，__init__.py要同步导入）
- atrumod_condition_evaluator.py  （条件分流评估器，__init__.py也要同步导入,未完全懂❌）
- yaml_config.py  (注册 ATRUMODConditionEvaluator 实例化通道，因为原来的rio-detr源代码仓库是硬编码的Evaluator，所以不实例化注册会报错)


```
#按条件分类的独立评测命令
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline.yml \
                --test-only \
                -r outputs/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline/best_stg1.pth \
                -u evaluator.type=ATRUMODConditionEvaluator
```

## 关于实验代码细节的地方

- 该加残差的加残差（网络不易过深）


```
运行训练命令
CUDA_VISIBLE_DEVICES=0 torchrun \
  --master_port=7001 \
  --nproc_per_node=1 \
  train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_dronevehicle_ir.yml \
  --use-amp \
  --seed=0


python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_prototype.yml --use-amp

```

我记得，在检测头的旋转框，旋转可变形注意力计算采样时，源代码有误

## 基线单模态热红外

```
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_single_ir.yml --use-amp
```

best score：epoch20，mAP=0.44100

## 基线单模态可见光

```
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_single_rgb.yml --use-amp
```

best score：epoch23，mAP=0.324482

## 基线拼接（S3、S4、S5）

```
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline.yml --use-amp
```

best score：epoch22，mAP=0.53416

## 基线直接相加（S3、S4、S5）

```
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml --use-amp
```

best score：epoch20，mAP=0.54038

## 模块一

- rtdetrv2_obb_decoder.py（rtv4，因为原版代码有点错误，修改了可变形注意力时的旋转采样）
- rtv4.py（rtv4，添加了透传训练时期的预测条件输出，为了送给criterion计算损失）
- rtv4_obb_criterion.py（rtv4，添加了条件损失映射，并且计算6维条件损失）
- dual_hgnetv2_lg_dense.py（backbone，模块一的网络结构代码,__init__.py也要同步导入）
- atrumod_dataset.py（atr-umod数据集加载器代码）
- profiler_utils.py（修改为了通过真实的配置文件中的输入尺寸分辨率来计算参数量、计算量等）
- det_engine.py（修改了字典也可以直接搬运到GPU上去做运算，因为我们的6维条件是由字典保存的，为满足计算预测的6维条件损失所做的改进）
- rtdetrv2_obb_hgnetv2_s_atrumod_prototype.yml（模块一的训练数据集、网络、损失等配置）
- inspect_atrumod.py（atr-umod数据集的探针代码，查看xml真实标注的信息）

best score：epoch23，mAP=0.52123

热红外与可见光的融合就是通过条件预测的向量经过线性和激活输出了权重，通过加权求和的方式进行的融合，但是热红外与可见光并没有互动起来。
也就是说，其实效果和基线肯定差不多。热红外与可见光的特征并没有互动起来。


## 验证模块一（消融模块一引入的显示条件先验到底有无作用）

```
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_self_driven.yml --use-amp

```

直接不实列化六个预测头，所以相当于没有显示条件损失的计算了

best score：epoch17，mAP=0.51557

所以这样引入显示条件先验失败了，该方案被我丢掉，我要重新取寻求好的方式引入显示条件先验服务双光融合检测


## InfraNet：

### InfraNet no make superior train's setting

对于Infranet论文的复现
- qual_gate.py  (质量感知融合模块，__init__.py同步导入该类，

code细节问题：

其中的q分数，是对可见光特征图的所有通道共享一个q；但每个热红外特征图是各个通道使用不同的权重；
但在质量感知融合时，无论是热红外还是可见光，都没有使用空间像素位置的权重，各通道空间像素位置共享同一个权重。
)

- dual_hgnetv2_infranet.py  (整个Infranet融合的backbone骨架，__init__.py也要同步导入)
- rtv4_infranet.py  (__init__.py要同步导入，专门为infranet而写的整个框架组装代码,未完全懂❌)
- rtv4_obb_criterion_infranet.py(__init__，专门为infranet而写的损失计算代码，未完全懂大❌)

- rtdetrv2_obb_hgnetv2_s_atrumod_infranet.yml （记得里面一定要显示的写
model: RTv4_Infranet，
criterion: RTv4OBBCriterion_Infranet这两行代码
）

- dual_hgnetv2_infranet_improve.py  (优化收敛速度的infranet整个backbone骨架代码，让可见光模态也使用了预训练权重，把hgnet拆开使用的，__init__.py也要同步导入)

- diagnose_infranet.py  (对infranet效果差的诊断分析)

```
#训练命令
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_infranet.yml --use-amp

```


```
#单模态热红外测试命令（因为InfraNet的网络优越性就是可以同时支持单模态与双光检测）
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_infranet.yml \
                --test-only \
                -r outputs/rtdetrv2_obb_hgnetv2_s_atrumod_infranet/best_stg1.pth \
                -u DualHGNetv2_InfraNet.eval_mode=ir

```
best score：epoch19，mAP=0.452196（双光）

输出文件：best_stg1_origin.pth、log_origin.txt


