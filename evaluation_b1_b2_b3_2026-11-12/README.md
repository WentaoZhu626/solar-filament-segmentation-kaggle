# 太阳暗条分割赛 B1 至 B3 交付记录

2026 年 11 月 11 日 · Solar Filament Segmentation Challenge 2026

使用仓库导出的同一批折外预测比较边界精修前后。数据覆盖 707 张训练原图、1,154 个标注视图和 8,199 个标注实例。预测表含 4,375 个实例，16 张无预测的观测也纳入计分。输入预测与 GitHub blob fe5044ee1ae84c452a4e7975b91e1c765ac48bf4 的内容校验一致。标注和样例原图来自 Kaggle 官方比赛训练数据。所有下载和分析均在本地完成。

| 任务 | 当前交付 | 文件 |
|---|---|---|
| B1 指标与分布 | 完成。按主办方自评 notebook 复算，并用 TorchMetrics 计算 Dice | [PQ 与各折结果](B1/csv/pq_by_fold.csv) · [Dice 与 IoU 摘要](B1/csv/summary.csv) · [直方图](B1/figures/dice_iou.png) · [拆分与合并分布](B1/figures/split_merge.png) |
| B2 原图展示 | 完成。四类共 9 个实例案例，附原图定位和各标注视图 | [样例总图](B2/figures/example_gallery.png) · [图号与证据](B2/csv/case_catalog.csv) · [标注分歧对照](B2/figures/09_annotator_disagreement.png) |
| B3 敏感性 | 已保存可复算的变体与仓库已有实验。完整流水线的若干单因素点待补导出 | [统一敏感性表](B3/csv/sensitivity_table.csv) · [固定候选与掩膜的扫描](B3/csv/fixed_pool_sweeps.csv) |

- [B1 指标工作簿](B1/B1_metrics.xlsx)的第一页并排比较精修前、精修后及变化量。
- [B1、B2、B3 汇总工作簿](B1_B2_B3_evaluation.xlsx)包含三个部分的表格。各目录的 CSV 保留机器可读的原始输出。
- B2 原图叠图在 [B2/figures](B2/figures)。用 [样例目录](B2/csv/case_catalog.csv)中的图号和文件名查找详图。
- B3 敏感性结果在 [B3/csv](B3/csv)，待补数据的 PQ 保持空白。

## B1 主办方指标

| 指标 | 精修前 | 精修后 |
|---|---:|---:|
| Pooled PQ | 0.448122 | **0.454131** |
| SQ，PQ 命中实例对的平均 IoU | 0.678334 | 0.681775 |
| RQ | 0.660621 | 0.666101 |
| TP / FP / FN | 5,063 / 2,066 / 3,136 | 5,105 / 2,024 / 3,094 |
| 整图前景 Dice，按标注视图取平均 | 0.658233 | 0.660640 |
| 有重叠实例对的 Dice 中位数 | 0.783835 | 0.787461 |
| 有重叠实例对的 IoU 中位数 | 0.644513 | 0.649432 |
| 有重叠的实例对数 | 6,565 | 6,598 |
| 与两个及以上预测相交的标注实例数 | 205 | 224 |
| 与两个及以上标注相交的预测实例视图数 | 217 | 246 |

- 每张观测分别与每位标注员的实例比较。用 pycocotools 将训练多边形栅格化并在 COCO RLE 上计算 IoU。主办方 notebook 将 IoU > 0.5 的每个实例对计为 TP，FP 和 FN 分别统计没有任何命中边的预测和标注。先累加全体视图的计数和命中 IoU，再计算 PQ。五折均提升，合并增量为 **+0.006010**。 [主办方自评 notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- Dice 与 IoU 直方图依照 notebook 纳入所有 IoU > 0 的实例对，各 50 个等宽区间。逐实例对 Dice 由 TorchMetrics 计算，并逐对校验 Dice = 2 × IoU / (1 + IoU)。整图前景 Dice 是额外指标，将每个视图的实例合并为前景再计算，每个视图权重相同。直方图纳入的实例对在精修前后各自确定，均值和中位数会同时受到掩膜变化与样本构成变化影响。
- 一对多、多对一用 IoU > 0 作为重叠边，逐实例记录度数，包括 0 和 1。预测重复出现在同一图的多个标注视图中，因此统计单位为实例视图。轻微接触也会增加度数，具体的合并和拆分错误需结合 B2 局部图判读。
- **仓库与自评 notebook 差 1 个 FP。** 在 20160810234034Lh.jpeg 的视图 010403 中，同一个预测和两个相互重叠的标注都超过 0.5，另有一个预测完全没有命中。仓库查看器用预测数减 TP 得到 0 个 FP，主办方 notebook 按未命中的预测列得到 1 个 FP。对全体矩阵运行 notebook 的 PQ 函数复得 0.44812152 / 0.45413138，另在精修前后各 15 个视图上用 notebook 的稠密掩膜算法核对了 Dice 和 IoU 矩阵。 [仓库查看器](https://github.com/WentaoZhu626/solar-filament-segmentation-kaggle/blob/main/tools/oof_viewer.py)

复核材料保存在 [输入清单与 SHA-256](B1/verification/run_manifest.json)、[官方函数交叉验证](B1/verification/official_notebook_check.json)、[逐视图计数](B1/csv/view_scores.csv)、[逐实例对数据](B1/csv/positive_pairs.csv)、[逐实例重叠度数](B1/csv/instance_degrees.csv)和[直方图区间表](B1/csv/histogram_bins.csv)。

## B2 展示样例

青色实线为标注，琥珀色虚线为精修预测。每张详图左侧为 2048 × 2048 Hα 原图和裁切位置，右侧同一裁切逐位展示标注员的轮廓。局部图使用各自灰度百分位拉伸，位置和掩膜保持原图像素坐标。图中标注员编号为该图内的视图顺序，原始视图 ID 见目录。

| 类别 | 图号 | 选例依据 |
|---|---|---|
| 做得好 | 01 | 预测与目标的 IoU 0.890，可见边界和细小突起的吻合情况 |
| 困难 | 02–04 | 409 px 小暗条，IoU 0.835。临边案例中心约在 0.905 个日盘半径处，IoU 0.643。弱候选案例来自低概率候选池，IoU 0.862 |
| 失败 | 05–08 | 零重叠漏检、各可用视图均无重叠的预测、一个预测覆盖两条标注、两段预测合计覆盖同一标注约 89.8% |
| 标注分歧 | 09 | 同一个预测与一位标注员的目标 IoU 为 0.843，与另一位标注员全部目标零重叠 |

04 的候选来源是弱概率分支。局部灰度对比另有量化记录，以候选内灰度中位数相对约 10 px 邻环中位数的降幅表示。05 的低对比漏检约为 4.7%，可作为低对比困难样例补充。09 的总图展示选中视图，详图展示两位标注员。 [九图总览](B2/figures/example_gallery.png) · [各图证据](B2/csv/case_catalog.csv)

## B3 六项敏感性

| 控制量 | 当前可引用的结果 | 完整单因素实验还缺什么 |
|---|---|---|
| 保留阈值 | 固定已导出掩膜，阈值 0.40 / 0.45 / 0.50 的精修后 PQ 为 0.45413 / 0.45171 / 0.44657 | 低于 0.40 需要未保留候选。要精确评估变化后精修结果，还需按新的保留集合重跑回贴与冲突处理 |
| CNN 混合权重 | 当前权重 0.60 的精修前 PQ 为 0.44812 | 同一候选的 GBM 与五个 CNN 折外分数，或各权重的整套折外掩膜 |
| 精修阈值 | 仓库记录同一 CNN 保留集下 0.40 / 0.50 / 0.60 的 PQ 为 0.4531 / 0.4542 / 0.4530 | 表中已标出仓库计分口径。若需要完全按本次 notebook 重算，补这三个阈值的 OOF RLE 或精修概率图及回贴坐标 |
| 主候选最小面积 | 固定现有候选和掩膜，按原始面积筛除 B 候选。400 / 450 / 500 px 的精修后 PQ 为 0.45413 / 0.45155 / 0.44712 | 改面积会影响弱候选可用区域和上下文。须按不同面积重建候选并重跑后续，再作为完整流水线结果 |
| 弱候选概率阈值 | 基线 p > 0.30。移除已保留的全部弱候选、固定其余掩膜时 PQ 由 0.45413 降至 0.44808 | p > 0.20 / 0.40 / 0.50 必须重新提取连通域并对其评分，不能从最终掩膜恢复 |
| 裁判数量 | 仓库记录五种单裁判混合方案的 PQ 为 0.4448–0.4475，五裁判方案记录为 0.4481 | 单裁判使用 CNN 权重 0.50，五裁判使用 0.60。控制其他参数时需补单裁判和 2–4 裁判的输出或候选分数 |

保留阈值复算使用 CSV 中存储的四位小数分数。对精修后掩膜的扫描固定已经完成的边界回贴和像素冲突处理。仓库历史实验保留其原始基准，避免把不同阈值、栅格化或筛选方式的差值合为一次单因素结果。 [精修实验记录](B3/source_notes/repository_BOUNDARY_REFINER.md) · [裁判实验记录](B3/source_notes/repository_CNN_JUDGE.md)

## 复跑与补数据

进入 [reproduce](reproduce) 后运行以下命令。复算使用本仓库的[最终折外预测](../results/oof_final_predictions.csv)和[折号](../results/folds.csv)，将两份 CSV 按原文件名放到 reproduce/inputs。另从 [Kaggle 比赛数据](https://www.kaggle.com/competitions/filament-segmentation-2026/data)取得训练标注 JSON，并按 [样例图像清单](B2/selection/image_list.txt)将 9 张训练原图放到 reproduce/inputs/train_images。官方自评 notebook 放到 reproduce/inputs/official，版本校验值见[复核记录](B1/verification/official_notebook_check.json)。复跑会在 reproduce 下生成结果。生成结果不需要重训。

~~~powershell
uv run --with-requirements requirements.txt python evaluate_oof.py
uv run --with-requirements requirements.txt python render_examples.py
uv run --with-requirements requirements.txt python sensitivity.py
uv run --with-requirements requirements.txt python verify_official.py
~~~

补 B3 时，最直接的交接方式是每个变体导出一个与 oof_final_predictions.csv 同结构的 CSV，使用相同 707 张图和同一折号，每次只改一个参数。没有预测的图无需占位。CSV 中的 rle_original、rle_refined 分别保留精修前后的 COCO RLE。每个变体另记下其他参数、模型权重版本及候选生成方式。在变体目录建立清单，prediction_csv 可用相对清单的路径。

~~~csv
variable,setting,prediction_csv
keep_threshold,0.35,keep_035.csv
cnn_weight,0.50,blend_050.csv
faint_threshold,0.40,faint_040.csv
~~~

~~~powershell
uv run --with-requirements requirements.txt python sensitivity.py --variants exports/manifest.csv
~~~

如果服务器侧先导出候选数据，保留阈值、混合权重和裁判数量的精修前对比可用一份全候选表。每行应包含观测文件名、折号、稳定候选 ID、来源、未精修 RLE、GBM 折外分数与五个裁判的折外分数，包含未过 0.40 的候选。精修后的可比结果仍需用对应保留集重新回贴。面积与弱候选阈值的全流水线实验需从 OOF 概率图重建候选，并沿用一致的折外训练与推理规则。
