from __future__ import annotations

import argparse
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

CHINESE_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]
CHINESE_FONT_FILES = [
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]
EN_STOPWORDS = set(ENGLISH_STOP_WORDS)

 
# 配置 matplotlib 使用的中文字体，避免图表中文显示乱码
def configure_chinese_font() -> None:
    # 优先加载 macOS 系统里常见的中文字体文件，避免图表标题乱码。
    for font_path in CHINESE_FONT_FILES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=str(font_path)).get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return

    # 如果没有找到字体文件，再退回到已注册字体名称。
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


# 解析并返回命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验四：垃圾邮件分类器文本分类设计与实现")
    parser.add_argument("--data", default="spam.csv", help="数据集路径")
    parser.add_argument("--text-col", default="v2", help="短信文本列名")
    parser.add_argument("--label-col", default="v1", help="标签列名")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--max-features", type=int, default=3000, help="词袋最大特征数")
    parser.add_argument(
        "--alphas",
        default="0.01,0.05,0.1,0.5,1.0",
        help="朴素贝叶斯平滑参数列表，使用英文逗号分隔",
    )
    return parser.parse_args()


# 将逗号分隔的 alpha 文本转换为浮点数列表
def parse_alpha_grid(alpha_text: str) -> list[float]:
    # 将命令行传入的 alpha 字符串拆分成浮点数列表，便于逐个训练模型。
    values = []
    for item in alpha_text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("至少需要提供一个 alpha 参数。")
    return values


# 从文件加载数据并返回只包含标签与文本的 DataFrame
def load_data(data_path: Path, label_col: str, text_col: str) -> pd.DataFrame:
    # 读取短信数据，并只保留标签列和文本列。
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据集：{data_path}")

    # 该数据集通常使用 latin-1 编码，这里按实验数据特点直接处理。
    df = pd.read_csv(data_path, encoding="latin-1")

    if label_col in df.columns and text_col in df.columns:
        df = df[[label_col, text_col]].copy()
    else:
        # 原始文件前两列是有效字段，后面的空列直接忽略。
        df = df.iloc[:, :2].copy()
        df.columns = [label_col, text_col]

    # 去掉缺失值，并把标签统一成小写文本，方便后续映射。
    df = df.dropna(subset=[label_col, text_col]).copy()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df[text_col] = df[text_col].astype(str)
    return df


# 清洗单条短信文本，移除网址与非字母字符
def clean_text(text: str) -> str:
    # 清洗短信内容：去网址、数字和符号，只保留英文单词。
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 为 DataFrame 添加 clean_text 列并移除空白样本
def add_clean_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    # 对每条短信执行清洗，并删除清洗后为空的样本。
    df = df.copy()
    df["clean_text"] = df[text_col].map(clean_text)
    df["clean_text"] = df["clean_text"].replace("", np.nan)
    df = df.dropna(subset=["clean_text"]).copy()
    return df


# 打印数据摘要并保存标签分布的图表
def show_dataset_overview(df: pd.DataFrame, label_col: str, text_col: str) -> None:
    # 输出数据概览，并保存类别分布图和饼图。
    print("\n1. 数据前 10 行：")
    print(df[[label_col, text_col]].head(10).to_string(index=False))

    # 统计正常邮件和垃圾邮件数量，便于查看样本是否平衡。
    print("\n2. 标签分布：")
    label_counts = df[label_col].value_counts().reindex(["ham", "spam"], fill_value=0)
    print(label_counts)

    # 先配置中文字体，再生成可视化结果。
    configure_chinese_font()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 绘制类别柱状图。
    plt.figure(figsize=(6, 4))
    counts = label_counts.reindex(["ham", "spam"], fill_value=0)
    plt.bar(["ham", "spam"], counts.values, color=["#4C78A8", "#F58518"])
    plt.title("短信类别数量分布")
    plt.xlabel("类别")
    plt.ylabel("数量")
    plt.tight_layout()
    bar_path = RESULTS_DIR / "label_distribution.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"\n3. 类别柱状图已保存：{bar_path}")

    # 绘制类别比例饼图。
    plt.figure(figsize=(6, 6))
    plt.pie(
        label_counts.values,
        labels=["正常邮件", "垃圾邮件"],
        autopct="%.1f%%",
        startangle=90,
        colors=["#4C78A8", "#F58518"],
    )
    plt.title("正常邮件与垃圾邮件比例")
    plt.tight_layout()
    pie_path = RESULTS_DIR / "label_pie_chart.png"
    plt.savefig(pie_path, dpi=200)
    plt.close()
    print(f"4. 类别饼图已保存：{pie_path}")


# 统计文本序列中的高频词，返回 top_n 列表
def get_top_words(texts: pd.Series, top_n: int = 30) -> list[tuple[str, int]]:
    # 统计某一类短信中的高频词，供后续绘图使用。
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in str(text).split() if token not in EN_STOPWORDS and len(token) > 1]
        counter.update(tokens)
    return counter.most_common(top_n)


# 将词频列表绘制为横向条形图并保存
def plot_top_words(words: list[tuple[str, int]], title: str, output_path: Path) -> None:
    # 将词频前 30 的结果画成横向条形图。
    if not words:
        return

    word_labels = [item[0] for item in words][::-1]
    word_counts = [item[1] for item in words][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(word_labels, word_counts, color="#4C78A8")
    plt.title(title)
    plt.xlabel("词频")
    plt.ylabel("词语")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# 分别统计正常邮件与垃圾邮件的高频词并保存图片
def analyze_word_frequency(df: pd.DataFrame, label_col: str) -> None:
    # 分别统计正常邮件和垃圾邮件的高频词，并保存词频图。
    ham_words = get_top_words(df.loc[df[label_col] == "ham", "clean_text"])
    spam_words = get_top_words(df.loc[df[label_col] == "spam", "clean_text"])

    print("\n5. 正常邮件高频词前 10 个：")
    print(pd.DataFrame(ham_words, columns=["词语", "词频"]).head(10).to_string(index=False))

    print("\n6. 垃圾邮件高频词前 10 个：")
    print(pd.DataFrame(spam_words, columns=["词语", "词频"]).head(10).to_string(index=False))

    plot_top_words(ham_words, "正常邮件高频词前 30 个", RESULTS_DIR / "ham_top_words.png")
    plot_top_words(spam_words, "垃圾邮件高频词前 30 个", RESULTS_DIR / "spam_top_words.png")
    print(f"\n7. 高频词图已保存：{RESULTS_DIR / 'ham_top_words.png'}")
    print(f"8. 高频词图已保存：{RESULTS_DIR / 'spam_top_words.png'}")


# 使用 CountVectorizer 构建词袋特征并返回 X, y, vectorizer
def build_features(df: pd.DataFrame, max_features: int) -> tuple[np.ndarray, np.ndarray, CountVectorizer]:
    # 使用词袋模型把文本转成数值特征。
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    x = vectorizer.fit_transform(df["clean_text"]).toarray()
    y = df["label_code"].to_numpy()
    return x, y, vectorizer


# 在给定 alpha 网格上训练并评估模型，返回指标表格
def evaluate_alpha_grid(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
) -> pd.DataFrame:
    # 逐个测试不同 alpha，并记录训练集和测试集指标。
    records = []

    for alpha in alphas:
        # 每个 alpha 训练一个朴素贝叶斯模型，比较对应结果。
        model = MultinomialNB(alpha=alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        precision = precision_score(y_test, test_pred, zero_division=0)
        recall = recall_score(y_test, test_pred, zero_division=0)

        records.append(
            {
                "alpha": alpha,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "precision": precision,
                "recall": recall,
            }
        )

    metrics_df = pd.DataFrame(records).sort_values(
        by=["train_accuracy", "test_accuracy", "alpha"], ascending=[False, False, True]
    ).reset_index(drop=True)
    return metrics_df


# 绘制并保存混淆矩阵图像
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    # 计算并保存混淆矩阵。
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    # 在格子中标出具体数值，便于观察分类错误分布。
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(col_index, row_index, f"{matrix[row_index, col_index]}", ha="center", va="center", color="black")
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])
    plt.colorbar()
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    plt.title("朴素贝叶斯分类混淆矩阵")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# 主运行流程：数据加载、训练、评估与结果保存
def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = DATA_DIR / data_path

    # 1. 读取数据并展示基础信息。
    df = load_data(data_path, args.label_col, args.text_col)
    show_dataset_overview(df, args.label_col, args.text_col)

    # 2. 清洗文本并查看清洗后的样例。
    df = add_clean_text(df, args.text_col)
    print("\n9. 清洗后前 5 行：")
    print(df[[args.label_col, args.text_col, "clean_text"]].head().to_string(index=False))

    # 3. 分别统计正常邮件和垃圾邮件的词频。
    analyze_word_frequency(df, args.label_col)

    # 4. 将标签映射成 0/1，方便模型训练。
    label_mapping = {"ham": 0, "spam": 1}
    df["label_code"] = df[args.label_col].map(label_mapping)
    if df["label_code"].isnull().any():
        unknown_labels = sorted(df.loc[df["label_code"].isnull(), args.label_col].unique())
        raise ValueError(f"发现无法识别的标签：{unknown_labels}")

    # 5. 构建词袋特征并输出特征维度。
    x, y, vectorizer = build_features(df, args.max_features)
    print(f"\n10. 词袋特征维度：{x.shape}")
    print(f"11. 词表大小：{len(vectorizer.get_feature_names_out())}")

    # 6. 按实验要求拆分训练集和测试集。
    x_train, x_test, y_train, y_test, train_index, test_index = train_test_split(
        x,
        y,
        df.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"\n12. 训练集数量：{len(x_train)}")
    print(f"13. 测试集数量：{len(x_test)}")

    # 7. 依次测试不同 alpha，记录评价结果并保存。
    alphas = parse_alpha_grid(args.alphas)
    metrics_df = evaluate_alpha_grid(x_train, x_test, y_train, y_test, alphas)
    metrics_path = RESULTS_DIR / "alpha_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n14. 不同 alpha 的模型评价结果：")
    print(metrics_df.to_string(index=False))
    print(f"\n15. alpha 评价结果已保存：{metrics_path}")

    # 8. 选取训练集准确率最高的模型重新训练并测试。
    best_row = metrics_df.iloc[0]
    best_alpha = float(best_row["alpha"])
    print(f"\n16. 训练集准确率最高的 alpha：{best_alpha}")

    best_model = MultinomialNB(alpha=best_alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\n17. 最优朴素贝叶斯模型测试结果：")
    print(f"准确率 accuracy：{accuracy:.4f}")
    print(f"精确率 precision：{precision:.4f}")
    print(f"召回率 recall：{recall:.4f}")

    # 9. 保存测试集预测对比结果，便于核对错误样本。
    compare_df = pd.DataFrame(
        {
            "真实值": y_test,
            "预测值": y_pred,
            "短信内容": df.loc[test_index, args.text_col].to_numpy(),
        }
    )
    compare_path = RESULTS_DIR / "prediction_compare.csv"
    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"\n18. 预测结果对比已保存：{compare_path}")

    # 10. 绘制混淆矩阵，观察模型在两类样本上的区分情况。
    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, confusion_path)
    print(f"19. 混淆矩阵已保存：{confusion_path}")


if __name__ == "__main__":
    main()