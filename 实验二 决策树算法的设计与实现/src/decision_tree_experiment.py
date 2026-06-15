import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 导入鸢尾花数据集，作为实验数据。
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# 定义目录结构：src 放代码，results 放结果，docs 放报告。
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


# 配置中文字体，避免决策树图片标题出现乱码或方框。
def configure_chinese_font():
    # 优先使用微软雅黑、黑体、宋体等 Windows 常见中文字体。
    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]

    # 找到可用字体后写入 Matplotlib 配置，并关闭负号乱码问题。
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return


# 自定义决策树节点类，用来保存每个节点的划分条件、子节点、样本分布和绘图坐标。
class TreeNode:

    def __init__(self, x_pos, y_pos, layer, class_labels):
        self.f = None  # 当前节点使用的切分特征名，叶子节点为 None
        self.v = None  # 当前节点使用的切分阈值，叶子节点为 None
        self.left = None # 左子节点，满足 f <= v 的样本会进入左子树
        self.right = None # 右子节点，满足 f > v 的样本会进入右子树
        self.pos = (x_pos, y_pos)  # NetworkX 绘图时使用的固定坐标
        self.label_dist = None  # 当前节点中各类别样本数量
        self.layer = layer # 当前节点所在层数，根节点为 1，后续层数递增
        self.class_labels = list(class_labels) # 类别标签列表，保证每个节点 label_dist 的类别含义一致

    # 定义节点打印格式：非叶子节点显示切分条件，叶子节点显示类别分布和样本数。
    def __str__(self):
        if self.f is not None:
            return f"{self.f}\n<= {self.v:.2f}"
        return f"{self.label_dist}\n({np.sum(self.label_dist)})"


# 加载数据集并整理成 DataFrame，方便后续处理和可视化。
def load_iris_dataframe():
    iris = load_iris()

    # 将特征数据转换为 DataFrame，并设置列名为原始特征名称。
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    # 将类别标签加入 DataFrame，便于查看完整表格。
    iris_df["target"] = iris.target

    # 按实验指导书要求，把列名改成更简洁的字段名。
    iris_df.columns = ["sepal_len", "sepal_width", "petal_len", "petal_width", "target"]
    return iris_df, iris.target_names


# 计算基尼不纯度，作为划分质量的评估指标。基尼不纯度越小，节点越纯。
def gini(y):
    # 空节点没有样本，不参与有效划分，直接返回 0。
    if len(y) == 0:
        return 0.0

    # value_counts() 统计每一类样本数量，再转换为类别比例。
    counts = y.value_counts()
    probs = counts / len(y)

    # Gini = 1 - 各类别比例平方和。
    return 1 - np.sum(np.square(probs))


# 递归生成决策树，保留每个节点的划分规则和样本分布信息，供后续预测和可视化使用。
def generate(X, y, x_pos, y_pos, nodes, min_leaf_samples, max_depth, layer, class_labels):
    # 创建当前节点，并计算该节点中各类别样本数量，保存在 label_dist 中，供后续预测时使用。
    current_node = TreeNode(x_pos, y_pos, layer, class_labels)
    current_node.label_dist = [int(len(y[y == v])) for v in class_labels]
    nodes.append(current_node)

    # 停止条件：样本太少、节点已经足够纯，或树的层数超过最大深度。
    if len(X) < min_leaf_samples or gini(y) < 0.1 or layer > max_depth:
        return current_node

    best_impurity_decrease, best_f, best_v = 0.0, None, None

    # 记录父节点的基尼不纯度，用来计算划分后的不纯度下降量。
    parent_gini = gini(y)

    # 遍历所有特征和候选切分点，保留不纯度下降量最大的划分。
    for f in X.columns:
        for v in sorted(X[f].unique()):
            # 按候选阈值 v 将样本划分为左右两个子集。
            left_mask = X[f] <= v
            right_mask = X[f] > v
            y1, y2 = y[left_mask], y[right_mask]

            # 两个子节点都必须达到最小样本数，避免生成过小叶子节点。
            if len(y1) >= min_leaf_samples and len(y2) >= min_leaf_samples:
                # 子节点 Gini 需要按左右子节点样本数量加权，样本越多影响越大。
                weighted_child_gini = (len(y1) / len(y)) * gini(y1) + (len(y2) / len(y)) * gini(y2)
                imp_descent = parent_gini - weighted_child_gini

                if imp_descent > best_impurity_decrease:
                    best_impurity_decrease, best_f, best_v = imp_descent, f, v

    # 将最佳切分特征和阈值保存到当前节点。
    current_node.f, current_node.v = best_f, best_v

    # 如果找到了有效切分点，就继续递归生成左右子树。
    if current_node.f is not None:
        left_mask = X[best_f] <= best_v
        right_mask = X[best_f] > best_v

        # 层数越深，水平偏移越小，使图中的子节点逐层收拢。
        offset = 2 ** max(max_depth - layer, 0)
        current_node.left = generate(
            X[left_mask],
            y[left_mask],
            x_pos - offset,
            y_pos - 1,
            nodes,
            min_leaf_samples,
            max_depth,
            layer + 1,
            class_labels,
        )
        current_node.right = generate(
            X[right_mask],
            y[right_mask],
            x_pos + offset,
            y_pos - 1,
            nodes,
            min_leaf_samples,
            max_depth,
            layer + 1,
            class_labels,
        )

    return current_node


# 训练自定义决策树，并返回根节点和所有节点列表，供后续预测和可视化使用。
def decision_tree_classifier(X, y, min_leaf_samples=5, max_depth=4):
    # nodes 用于保存整棵树的所有节点，后续可视化时会用到。
    nodes = []

    # 固定类别顺序，保证每个节点 label_dist 的类别含义一致。
    class_labels = sorted(y.unique())

    # 从根节点开始递归生成整棵决策树。
    root = generate(
        X,
        y,
        0,
        0,
        nodes,
        min_leaf_samples=min_leaf_samples,
        max_depth=max_depth,
        layer=1,
        class_labels=class_labels,
    )
    return root, nodes


# 从根节点向下查找，直到叶子节点，完成单条样本预测。
def predict_one(root, row):
    node = root

    # 非叶子节点有切分特征，根据阈值判断进入左子树还是右子树。
    while node.f is not None:
        if row[node.f] <= node.v:
            node = node.left
        else:
            node = node.right

    # 到达叶子节点后，选择该节点中样本数量最多的类别作为预测结果。
    return node.class_labels[int(np.argmax(node.label_dist))]


# 对 DataFrame 中的所有样本逐条预测。
def predict(root, X):
    return np.array([predict_one(root, row) for _, row in X.iterrows()])


# 将自定义树结构转换为 NetworkX 有向图，供后续绘图使用。
def get_networkx_graph(G, root):
    # 左子节点存在时，添加当前节点到左子节点的边，并继续递归。
    if root.left is not None:
        G.add_edge(root, root.left)
        get_networkx_graph(G, root.left)

    # 右子节点存在时，添加当前节点到右子节点的边，并继续递归。
    if root.right is not None:
        G.add_edge(root, root.right)
        get_networkx_graph(G, root.right)
    return G


# 读取每个节点预先保存的坐标，供 NetworkX 绘图使用。
def get_tree_pos(G):
    return {node: node.pos for node in G.nodes}


# 设置节点颜色：内部节点为灰色，叶子节点按预测类别区分。
def get_node_color(G):
    color_dict = []
    for node in G.nodes:
        # 叶子节点没有切分特征，用类别分布中最多的类别决定颜色。
        if node.f is None:
            label = int(np.argmax(node.label_dist))
            if label % 3 == 0:
                color_dict.append("#007979")
            elif label % 3 == 1:
                color_dict.append("#E4007F")
            else:
                color_dict.append("blue")
        else:
            # 非叶子节点统一使用灰色，表示该节点只是判断条件。
            color_dict.append("gray")
    return color_dict


# 绘制并保存训练得到的决策树可视化图片。
def draw_tree(root, output_path, title):
    # 先把自定义树转换成 NetworkX 图结构。
    G = nx.DiGraph()
    G.add_node(root)
    get_networkx_graph(G, root)

    # 获取节点坐标、节点文字标签和节点颜色。
    pos = get_tree_pos(G)
    labels = {node: str(node) for node in G.nodes}
    colors = get_node_color(G)

    # 使用 NetworkX 绘制决策树，并保存为图片文件。
    plt.figure(figsize=(13, 7))
    ax = plt.gca()
    nx.draw_networkx(
        G,
        pos=pos,
        ax=ax,
        labels=labels,
        node_shape="o",
        font_color="white",
        node_size=3000,
        node_color=colors,
        arrows=True,
        arrowsize=16,
        linewidths=1.2,
        edge_color="#555555",
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# 主函数，负责整体流程控制：加载数据、训练模型、评估性能、保存结果和可视化。
def main():
    # 先配置中文字体，保证后续图表标题能正常显示中文。
    configure_chinese_font()

    # 解析命令行参数，结果标题仅保留实验名称。
    parser = argparse.ArgumentParser(description="实验二：决策树算法的设计与实现")
    parser.add_argument("--min-leaf-samples", type=int, default=5, help="叶子节点最少样本数量")
    parser.add_argument("--max-depth", type=int, default=4, help="决策树最大深度")
    args = parser.parse_args()

    # 加载鸢尾花数据，并拆分出特征矩阵 X 和标签 y。
    iris_df, target_names = load_iris_dataframe()
    X_iris = iris_df.iloc[:, :-1]
    y_iris = iris_df["target"]

    # 保留数据前 5 行、缺失值和类别分布等探索性输出。
    print("鸢尾花数据集前5行：")
    print(iris_df.head())
    print("\n数据集形状：", iris_df.shape)
    print("\n缺失值统计：")
    print(iris_df.isnull().sum())
    print("\n类别分布：")
    print(y_iris.value_counts().sort_index())
    print("\n各类别名称：")
    for i, name in enumerate(target_names):
        print(f"{i}: {name}")

    # 按 8:2 划分训练集和测试集，stratify 用于保持各类别比例一致。
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris,
        y_iris,
        test_size=0.2,
        random_state=42,
        stratify=y_iris,
    )

    # 训练手写决策树
    root, nodes = decision_tree_classifier(
        X_train,
        y_train,
        min_leaf_samples=args.min_leaf_samples,
        max_depth=args.max_depth,
    )
    y_pred = predict(root, X_test)

    # 计算模型评价指标，作为实验结果写入控制台和结果文件。
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    matrix = confusion_matrix(y_test, y_pred)

    # 输出训练与测试规模、节点数量和分类性能。
    print("\n训练集样本数：", len(X_train))
    print("测试集样本数：", len(X_test))
    print("决策树节点数：", len(nodes))
    print(f"准确率 accuracy：{accuracy:.4f}")
    print("\n分类报告：")
    print(report)
    print("混淆矩阵：")
    print(matrix)

    # 绘制并保存决策树结构图。
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tree_path = RESULTS_DIR / "decision_tree_visualization.png"
    draw_tree(root, tree_path, "决策树分类模型")

    # 保存逐条预测结果，便于报告中追溯真实类别和预测类别。
    compare_df = X_test.copy()
    compare_df["真实类别"] = y_test.values
    compare_df["预测类别"] = y_pred
    compare_df["真实类别名称"] = [target_names[i] for i in y_test.values]
    compare_df["预测类别名称"] = [target_names[i] for i in y_pred]
    compare_df.to_csv(RESULTS_DIR / "decision_tree_prediction_compare.csv", index=False, encoding="utf-8-sig")

    # 单独保存核心指标，便于快速查看，也方便后续生成报告时复用。
    metrics_df = pd.DataFrame(
        [
            ["accuracy", accuracy],
            ["node_count", len(nodes)],
            ["train_samples", len(X_train)],
            ["test_samples", len(X_test)],
            ["min_leaf_samples", args.min_leaf_samples],
            ["max_depth", args.max_depth],
        ],
        columns=["metric", "value"],
    )
    metrics_df.to_csv(RESULTS_DIR / "decision_tree_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"\n决策树可视化已保存：{tree_path}")
    print("预测对比已保存：decision_tree_prediction_compare.csv")
    print("指标文件已保存：decision_tree_metrics.csv")


if __name__ == "__main__":
    main()
