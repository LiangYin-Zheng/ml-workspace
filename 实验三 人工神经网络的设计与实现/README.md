# 实验三：人工神经网络的设计与实现

本实验基于鸢尾花数据集实现感知机分类，完成数据读取、标签统计、手写感知机训练、sklearn 感知机对照训练、数据打乱后的重新训练，以及结果可视化和对比输出。

## 文件说明

- `src/perceptron_experiment.py`：实验主程序，负责数据加载、训练、评估和绘图。
- `docs/实验三 人工神经网络的设计与实现.pdf`：实验指导书。
- `docs/实验三 人工神经网络的设计与实现.docx`：实验文档。
- `docs/实验三_人工神经网络_内容分析.md`：指导书要求与实现情况对照。
- `results/custom_perceptron.png`：手写感知机分类结果图。
- `results/sklearn_perceptron.png`：sklearn 感知机分类结果图。
- `results/shuffle_perceptron.png`：数据打乱后重新训练的分类结果图。
- `results/prediction_compare.csv`：真实值与预测值对比结果。
- `requirement.txt`：运行依赖说明。

## 数据说明

程序直接调用 `sklearn.datasets.load_iris()` 加载鸢尾花数据集，不需要额外准备数据文件。运行时会先输出全量数据的行列信息和各类标签数量，然后取前 70 条样本参与感知机实验。

当前脚本在二分类阶段使用前 70 条样本中的前两个特征：

| 字段 | 含义 |
|---|---|
| `sepal_len` | 花萼长度 |
| `sepal_width` | 花萼宽度 |
| `petal_len` | 花瓣长度 |
| `petal_width` | 花瓣宽度 |
| `target` | 原始类别标签 |

二分类时将 `target == 0` 作为一类，其余样本作为另一类。

## 运行方式

建议在实验目录下执行：

```powershell
cd ".\实验三 人工神经网络的设计与实现"
python .\src\perceptron_experiment.py
```

可选参数：

```powershell
python .\src\perceptron_experiment.py --epochs 200 --lr 0.1 --test-size 0.2
```

## 输出结果

程序运行后会生成：

- `results/custom_perceptron.png`：手写感知机分类结果。
- `results/sklearn_perceptron.png`：sklearn 感知机分类结果。
- `results/shuffle_perceptron.png`：打乱数据后重新训练的分类结果。
- `results/prediction_compare.csv`：样本真实值、手写感知机预测值和 sklearn 预测值对比。
- 控制台输出：数据行列、完整数据表、各类标签数量、前 70 行样本表、手写感知机参数、sklearn 感知机参数以及准确率。

## 注意事项

1. 该实验脚本默认使用鸢尾花数据集，不依赖额外 CSV 文件。
2. 结果文件会输出到 `results/` 目录，运行前无需手动创建该目录。
3. 代码已经配置常见中文字体，便于直接生成中文图标题。
4. 当前脚本只生成 `prediction_compare.csv`，没有单独输出 `metrics.csv`。
