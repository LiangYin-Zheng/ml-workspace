# 实验一：贝叶斯分类器设计与实现

本实验完成中文外卖评论情感二分类，流程包括数据读取、缺失值与重复值检查、中文文本清洗、TF-IDF 向量化、GaussianNB 训练与评价，以及结果保存。

## 文件说明

- `src/bayes_classifier_experiment.py`：实验主程序。
- `docs/实验一_贝叶斯分类器_内容分析.md`：指导书要求与实现情况对照。
- `docs/实验一 贝叶斯分类器设计与实现.pdf`：实验指导书。
- `docs/实验一 贝叶斯分类器设计与实现-实验报告.docx`：实验报告。
- `results/label_distribution.png`：情感类别分布图。
- `results/prediction_compare.csv`：测试集真实值与预测值对比。
- `data/takeout_reviews.csv`：中文外卖评论数据集。
- `data/hit_stopwords.txt`：哈工大停用词表。
- `requirements.txt`：运行依赖。

## 数据格式

默认读取 `data/takeout_reviews.csv`，要求包含以下两列：

| 列名 | 含义 |
|---|---|
| `review` | 外卖评论文本 |
| `label` | 情感标签，`0` 表示消极，`1` 表示积极 |

默认停用词文件为 `data/hit_stopwords.txt`。

## 运行方式

建议在实验目录下执行：

```powershell
cd ".\实验一 贝叶斯分类器设计与实现"
python .\src\bayes_classifier_experiment.py
```

如果数据文件或列名不同，可以显式指定：

```powershell
python .\src\bayes_classifier_experiment.py --data "你的数据.csv" --stopwords "你的停用词表.txt" --text-col "评论列名" --label-col "标签列名"
```

## 输出结果

程序运行后会生成：

- `results/label_distribution.png`：情感类别分布图。
- `results/prediction_compare.csv`：测试集评论、真实值和预测值对比。
- 控制台输出：数据概览、缺失值统计、重复值数量、清洗后样例，以及 accuracy、precision、recall、F1。

## 注意事项

1. 请先把 `takeout_reviews.csv` 和 `hit_stopwords.txt` 放入 `data/` 目录，或者使用参数指定绝对路径。
2. 该脚本内部会按项目目录解析数据与结果路径，因此也可以从其他工作目录运行。
3. 由于 `GaussianNB` 需要稠密矩阵，代码会把 TF-IDF 结果转换为数组后再训练。
4. 如需改用其他数据文件或列名，请通过参数显式指定。
