# 实验二：决策树算法的设计与实现

本实验在鸢尾花数据集上手动实现分类决策树，完成数据读取、基尼系数计算、递归生成决策树、预测、评价和可视化。

## 文件说明

- `src/decision_tree_experiment.py`：实验主程序，负责数据读取、树构建、预测和可视化。
- `docs/实验二_决策树算法_内容分析.md`：指导书要求与实现情况对照。
- `docs/实验二 决策树算法的设计与实现.pdf`：实验指导书。
- `docs/实验二决策树算法的设计与实现实验报告.docx`：实验报告。
- `results/decision_tree_visualization.png`：决策树可视化图片。
- `results/decision_tree_prediction_compare.csv`：测试集特征、真实类别和预测类别对比。
- `results/decision_tree_metrics.csv`：实验指标文件。
- `requirements.txt`：运行依赖。

## 数据说明

本实验直接使用 `sklearn.datasets.load_iris()` 加载鸢尾花数据集，不需要额外准备 CSV 文件。程序会把原始特征整理为以下字段：

| 列名 | 含义 |
|---|---|
| `sepal_len` | 花萼长度 |
| `sepal_width` | 花萼宽度 |
| `petal_len` | 花瓣长度 |
| `petal_width` | 花瓣宽度 |
| `target` | 鸢尾花类别标签，取值为 `0`、`1`、`2` |

## 运行方式

建议在实验目录下执行：

```powershell
cd ".\实验二 决策树算法的设计与实现"
python .\src\decision_tree_experiment.py
```

如需调整叶子节点最少样本数或最大深度：

```powershell
python .\src\decision_tree_experiment.py --min-leaf-samples 5 --max-depth 4
```

## 输出结果

程序运行后会生成：

- `results/decision_tree_visualization.png`：决策树模型可视化图。
- `results/decision_tree_prediction_compare.csv`：测试集特征、真实类别、预测类别及名称对比。
- `results/decision_tree_metrics.csv`：accuracy、节点数、训练集样本数、测试集样本数和参数记录。
- 控制台输出：鸢尾花数据前 5 行、缺失值统计、类别分布、分类报告和混淆矩阵。

## 注意事项

1. 本实验按指导书要求手动实现决策树节点、基尼系数计算和递归生成过程，没有直接调用 `sklearn.tree.DecisionTreeClassifier`。
2. `sklearn` 仅用于加载数据集、划分训练测试集和计算评价指标。
3. 如果调整 `min_leaf_samples`、`max_depth` 或随机种子，树结构和结果指标会发生变化。
4. 如需调整叶子节点最少样本数或最大深度，可直接通过参数修改。
