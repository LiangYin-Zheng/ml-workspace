import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron # sklearn 的感知机模型，提供了更稳定和高效的训练算法
from sklearn.metrics import accuracy_score # 用于评估模型性能的准确率指标
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# 优先加载 Windows 常见中文字体，避免图标题乱码。
def setup_font():
    for p in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        path = Path(p)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(path)).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return

# 加载 Iris 数据集并转换为 DataFrame 格式，方便后续处理和展示。
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["sepal_len", "sepal_width", "petal_len", "petal_width"])
    df["target"] = iris.target
    return df

# 手写感知机训练函数，使用简单的误分类更新规则，返回权重和偏置。
def perceptron_train(x, y, lr=0.1, epochs=50):
    # 手写感知机，y 只保留 0/1 两类，内部映射到 -1/+1。
    w = np.zeros(x.shape[1], dtype=float)   # 权重初始化为 0
    b = 0.0                                 # 偏置初始化为 0
    y2 = np.where(y == 1, 1, -1)            # 将标签转换为 -1/+1，便于感知机更新规则
    for _ in range(epochs):                 # 迭代训练，最多 epochs 次
        errors = 0
        for xi, yi in zip(x, y2):
            if yi * (xi @ w + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
                errors += 1
        if errors == 0:
            break
    return w, b

# 预测函数，输出 0/1 类别。
def predict(x, w, b):
    return ((x @ w + b) >= 0).astype(int)

# 绘制数据点和决策边界的函数，支持显示间隔线。
def draw(x, y, w, b, title, path, show_margin=False):
    plt.figure(figsize=(6.8, 5.2))
    names = {0: ("setosa", "#1f77b4"), 1: ("versicolor", "#ff7f0e")}
    for label, (name, color) in names.items():
        m = y == label
        plt.scatter(x[m, 0], x[m, 1], s=52, c=color, edgecolors="white", label=name)

    xs = np.linspace(x[:, 0].min() - 0.5, x[:, 0].max() + 0.5, 200)
    if abs(w[1]) > 1e-12:
        ys = -(w[0] * xs + b) / w[1]
        plt.plot(xs, ys, c="#222222", lw=2, label="decision boundary")
        if show_margin:
            plt.plot(xs, -(w[0] * xs + b - 1) / w[1], c="#777777", ls="--", lw=1)
            plt.plot(xs, -(w[0] * xs + b + 1) / w[1], c="#777777", ls="--", lw=1)
    else:
        plt.axvline(-b / w[0], c="#222222", lw=2, label="decision boundary")

    plt.title(title)
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    # 解析命令行参数，设置学习率、训练轮数和测试集比例。
    parser = argparse.ArgumentParser(description="实验三 - 人工神经网络(感知机)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    # 设置字体和创建结果目录，加载数据并展示基本信息。
    setup_font()
    RESULTS.mkdir(exist_ok=True)

    # 加载 Iris 数据集并展示完整数据表和标签分布情况。
    df = load_data()
    full_table = df[["sepal_len", "sepal_width", "petal_len", "petal_width", "target"]].copy()
    full_table["label_name"] = full_table["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    print("数据行列:", df.shape)
    print("\n完整数据表:")
    print(full_table)
    print("\n各类标签数量:")
    print(full_table["label_name"].value_counts().reindex(["setosa", "versicolor", "virginica"], fill_value=0))

    # 取前 70 行，按实验要求用前两个特征做二分类：0 为 setosa，1 为 versicolor/virginica。
    part = df.iloc[:70, [0, 1, 4]].copy()
    part["label_name"] = part["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    part["binary_label"] = (part["target"] != 0).astype(int)
    x = part[["sepal_len", "sepal_width"]].to_numpy()
    y = part["binary_label"].to_numpy()

    print("\n前 70 行、列 0/1/最后一列:")
    print(part)
    print("\n前 70 行各类数量:")
    print(part["label_name"].value_counts().reindex(["setosa", "versicolor", "virginica"], fill_value=0))

    # 训练手写感知机模型并评估性能。
    w, b = perceptron_train(x, y, lr=args.lr, epochs=args.epochs)
    pred = predict(x, w, b)
    acc = accuracy_score(y, pred)
    print("\n手写感知机:", "w=", w, "b=", b, "acc=", round(acc, 4))
    draw(x, y, w, b, "自定义感知机模型训练结果", RESULTS / "custom_perceptron.png", show_margin=True)

    # 使用 sklearn 的 Perceptron 模型进行训练和评估，比较结果。
    model = Perceptron(max_iter=args.epochs, eta0=args.lr, tol=1e-4, random_state=42)
    model.fit(x, y) # 训练模型
    sk_pred = model.predict(x)
    sk_acc = accuracy_score(y, sk_pred) # 评估模型性能
    print("sklearn 感知机:", "w=", model.coef_[0], "b=", model.intercept_[0], "acc=", round(sk_acc, 4))
    draw(x, y, model.coef_[0], model.intercept_[0], "sklearn 感知机分类结果", RESULTS / "sklearn_perceptron.png", show_margin=True)

    # 数据混洗后再分割，按实验要求做修正结果。
    mix = part.sample(frac=1.0, random_state=42).reset_index(drop=True)
    mx = mix[["sepal_len", "sepal_width"]].to_numpy()
    my = mix["binary_label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(mx, my, test_size=args.test_size, shuffle=False) # 不再混洗，保持原顺序

    w2, b2 = perceptron_train(x_train, y_train, lr=args.lr, epochs=args.epochs)
    pred2 = predict(x_test, w2, b2)
    acc2 = accuracy_score(y_test, pred2)
    print("修正后手写感知机:", "w=", w2, "b=", b2, "acc=", round(acc2, 4))
    draw(x_test, y_test, w2, b2, "修正后的分类结果", RESULTS / "shuffle_perceptron.png", show_margin=True)

    out = pd.DataFrame({
        "sepal_len": x[:, 0],
        "sepal_width": x[:, 1],
        "y_true": y,
        "custom_pred": pred,
        "sklearn_pred": sk_pred,
    })
    out.to_csv(RESULTS / "prediction_compare.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
