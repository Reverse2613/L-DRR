# ATR-UMOD：双流 YOLO11s-OBB 条件融合原型

本目录独立接入 Ultralytics 8.3.0，不导入或修改原 engine 注册树。基线为**双骨干 YOLO11s-OBB＋Concat→1×1投影**，不是单流官方模型的原样成绩。两个骨干独立学习，共享原生 neck 和11类 OBB head。

先读 [config.py](config.py)、[data.py](data.py)，再读 [conditions.py](conditions.py)、[fusion.py](fusion.py)、[model.py](model.py)。训练、验证与几何复评分别在 [trainer.py](trainer.py)、[validator.py](validator.py)、[evaluation.py](evaluation.py)。

```text
IR/RGB [B,3,512,640] → 两个 YOLO11s 骨干 → indices 4/6/10
                   → stride 8/16/32，[256,256,512]通道工作投影/GN
高度/倾角 → 5维描述符 → RGB→IR微校准（仅重采样RGB）
天气/光照/场所 → 冻结原始CLIP512 → FiLM → CG-DMF
                   → 共享原生neck → OBB检测头
```

条件“解耦”指直接输入路径分工，不宣称统计正交：环境向量不编码高度、倾角或时间；几何描述符不读取环境标签。几何重采样后的视觉特征仍参与评分。没有条件预测损失、旧双向消息、联合增益门、混洗分支或几何MLP。

沿用已有公式：偏移/FiLM/评分读出零初始化，Query与隐藏卷积常规初始化；三尺度每坐标分量最多12个输入像素，高度/倾角调节经验包络。FP32差分双线性采样在零偏移时精确保持输入并保留梯度；包络不是经标定证明的物理视差。

| 配置后缀（configs/yolo_rgbt/atrumod_yolo11s_*.yml） | 作用 |
| --- | --- |
| concat | 简单双流基线，不使用条件 |
| condition | CLIP＋FiLM＋CG-DMF，无微校准 |
| condition_align | 完整请求：几何微校准＋FiLM＋CG-DMF |
| constant | 同完整结构，环境固定缓存第0行；几何仍真实 |
| no_film / equal_fusion | 分别绕过FiLM / 固定等权，保留公共参数初始化 |
| visual_score | 保留FiLM/视觉评分，去掉评分里的直接条件项 |
| visual_align | 无遥测描述符，视觉偏移使用全局12像素界限 |

visual_align目前用于ATR消融；本轮没有实现DroneVehicle/VLM数据接入。

数据直接读取原始成对图像与labels_ir。所有输入为512×640，六通道顺序为IR三通道、RGB三通道。不补默认条件、不随机换图、不做随机增强、不用1120补边，也不隐式缩小目标。有效框的中心/角点越界不导致删除。

解析阶段沿用边长≤1像素的退化框过滤，非有限退化框另记审计。原生损失另过滤边长<2像素的训练目标，记录数量/类别；评测仍使用全部解析后原始GT。[loss.py](loss.py)仅修正8.3.0非方形输入的小框宽高单位及有效GT识别；原生ProbIoU＋BCE＋DFL/分配器不变，所有基线和消融共用。

770×512 CLIP缓存核验模板、原始ID、来源与SHA256。它不进入EMA的state_dict更新；整体模型保存仍包含缓存，half/float保持FP32原值。预训练加载显式复制双骨干与兼容neck/head，只有新融合和不同类别输出卷积可以不匹配；错误模型/大小/不明缺键报错。

原型配方：36轮、训练/验证batch8、无增强、AMP/EMA、AdamW。预训练参数LR=1e-4，新参数LR=1e-3；矩阵WD=1e-4，Norm/bias/embedding/标量WD=0；warmup500迭代＋cosine至0.1倍，实际有效batch8，clip10。绕过原生AMP辅助检测器下载，改用本地检查；完整训练仍由你执行。

训练通过 [train_yolo_rgbt.py](../tools/train_yolo_rgbt.py)，每轮按**原生ProbIoU＋101点积分AP50**选best.pt；CSV的epoch从1开始。之后运行 [eval_yolo_rgbt.py](../tools/eval_yolo_rgbt.py)，导出未裁剪原图像素/radian预测，并调用原仓库不改动的几何IoU/VOC07评测。AP50_95是VOC07十阈值均值；两套AP不能混列。predictions.json可直接复评，省去再次推理。

输出含experiment.json、预训练覆盖、数据审计、参数组、results.csv与best.pt/last.pt。已有训练目录拒绝复用；resume只用于同实验中断续跑。原型沿用已经参与开发的官方test，不称新未接触测试集。

[check_yolo_rgbt.py](../tools/check_yolo_rgbt.py)限540秒，验证真实批次前后向与native trainer接口，检测优化步为0。缺少本地预训练时只验未训练结构，不能当真实权重/AP验收。安装、手动下载、检查和训练命令见 [flow.md](../plan/flow.md)。

基础来源：[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)、[CLIP](https://proceedings.mlr.press/v139/radford21a.html)、[FiLM](https://arxiv.org/abs/1709.07871)、[Spatial Transformer](https://arxiv.org/abs/1506.02025)、[C2Former](https://arxiv.org/html/2306.16175v3)。native接口/损失适配保留Ultralytics的AGPL-3.0声明；这些基础算子不分别主张首创。

