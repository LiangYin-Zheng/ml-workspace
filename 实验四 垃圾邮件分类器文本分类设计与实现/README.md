# 实验四：垃圾邮件分类器文本分类设计与实现

本实验完成英文短信垃圾邮件分类，主要流程包括数据读取、文本清洗、词频统计、词袋模型构建、朴素贝叶斯分类、参数对比和混淆矩阵绘制。

## 文件说明

- `src/spam_classifier_experiment.py`：实验主程序。
- `data/spam.csv`：短信分类数据集。
- `results/label_distribution.png`：正常邮件与垃圾邮件数量分布图。
- `results/label_pie_chart.png`：正常邮件与垃圾邮件比例饼图。
- `results/ham_top_words.png`：正常邮件高频词统计图。
- `results/spam_top_words.png`：垃圾邮件高频词统计图。
- `results/alpha_metrics.csv`：不同平滑参数 `alpha` 的模型评价结果。
- `results/prediction_compare.csv`：测试集真实值与预测值对比。
- `results/confusion_matrix.png`：朴素贝叶斯分类混淆矩阵。
- `requirement.txt`：运行依赖。

## 数据格式

默认读取 `data/spam.csv`，原始数据前两列分别为：

| 列名 | 含义 |
|---|---|
| `v1` | 标签，`ham` 表示正常邮件，`spam` 表示垃圾邮件 |
| `v2` | 短信文本内容 |

程序会自动忽略多余空列，并将文本统一转成小写后再进行清洗与建模。

## 运行方式

建议在实验目录下执行：

```powershell
cd ".\实验四 垃圾邮件分类器文本分类设计与实现"
python .\src\spam_classifier_experiment.py
```

如果数据文件名或列名不同，可以显式指定：

```powershell
python .\src\spam_classifier_experiment.py --data "你的数据.csv" --text-col "文本列名" --label-col "标签列名"
```

可选参数：

- `--test-size`：测试集比例，默认 `0.2`。
- `--random-state`：随机种子，默认 `42`。
- `--max-features`：词袋最大特征数，默认 `3000`。
- `--alphas`：朴素贝叶斯平滑参数列表，默认 `0.01,0.05,0.1,0.5,1.0`。

## 实验流程

1. 读取短信数据并展示前 10 行。
2. 统计正常邮件和垃圾邮件数量，并保存柱状图和饼图。
3. 清洗文本内容，去除数字、网址和符号。
4. 分别统计正常邮件和垃圾邮件的高频词，并保存前 30 个词频图。
5. 使用词袋模型将文本转换为特征向量。
6. 按不同 `alpha` 训练多项式朴素贝叶斯模型，比较训练准确率、测试准确率、精确率和召回率。
7. 选取训练集准确率最高的模型，输出测试集结果并绘制混淆矩阵。

## 输出结果

程序运行后会在 `results/` 目录生成以下文件：

- `label_distribution.png`
- `label_pie_chart.png`
- `ham_top_words.png`
- `spam_top_words.png`
- `alpha_metrics.csv`
- `prediction_compare.csv`
- `confusion_matrix.png`

控制台会输出数据概览、词频统计、模型评价结果，以及 `accuracy`、`precision`、`recall` 等指标。

## 依赖安装

如需单独安装依赖，可以执行：

```powershell
pip install -r .\requirement.txt
```

## 注意事项

1. 该数据集通常使用 `latin-1` 编码，脚本已内置处理。
2. 运行时会自动创建 `results/` 目录。
3. 如果中文图表字体显示异常，请确认系统是否存在可用的中文字体文件。

