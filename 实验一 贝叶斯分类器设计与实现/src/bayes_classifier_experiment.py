from __future__ import annotations

import argparse
import re
from pathlib import Path

import jieba
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


DEFAULT_SYMBOLS = r"-\n～%≥℃|/【】↓#~_「♂!？'，、:；。《》()（）·—.…,0123456789abcdefghijklnmopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHINESE_FONT_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]


def configure_chinese_font() -> None:
    """Configure a Chinese-capable font for matplotlib charts."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in CHINESE_FONT_CANDIDATES:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_stopwords(stopwords_path: Path) -> set[str]:
    """加载哈工大停用词表"""
    if not stopwords_path.exists():
        raise FileNotFoundError(
            f"未找到停用词表：{stopwords_path}\n"
            "请将哈工大停用词表放到当前目录，并命名为 hit_stopwords.txt。"
        )

    with stopwords_path.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def data_cleaning(content_list: list[str], stopwords: set[str]) -> list[str]:
    """按实验要求清洗评论文本：去特殊字符、分词、去停用词。"""
    content_seg = []
    symbols_pattern = "[" + re.escape(DEFAULT_SYMBOLS) + "]"

    for content in content_list:
        content = str(content)
        content = re.sub(symbols_pattern, " ", content)
        con_list = jieba.cut(content, cut_all=False)
        result_list = []

        for con in con_list:
            con = con.strip()
            if con and con not in stopwords and con != "\n" and con != "\u3000" and con != " ":
                result_list.append(con)

        str1 = " ".join(result_list)
        content_seg.append(str1)

    return content_seg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实验一：高斯贝叶斯中文外卖评论情感分类")
    parser.add_argument("--data", default="takeout_reviews.csv", help="CSV 数据集路径")
    parser.add_argument("--stopwords", default="hit_stopwords.txt", help="哈工大停用词表路径")
    parser.add_argument("--text-col", default="review", help="评论文本列名")
    parser.add_argument("--label-col", default="label", help="情感标签列名")
    parser.add_argument("--max-features", type=int, default=3000, help="TF-IDF 词向量最大维度")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(__file__).resolve().parent
    project_dir = work_dir.parent
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"
    data_path = Path(args.data)
    stopwords_path = Path(args.stopwords)

    if not data_path.is_absolute():
        data_path = data_dir / data_path
    if not stopwords_path.is_absolute():
        stopwords_path = data_dir / stopwords_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"未找到数据集：{data_path}\n"
            "请将中文外卖评论 CSV 数据集放到当前目录，或使用 --data 指定路径。"
        )

    df = pd.read_csv(data_path)
    print("\n1. 数据前 5 行：")
    print(df.head())

    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise ValueError(
            f"列名不匹配。当前数据列为：{list(df.columns)}\n"
            f"请确认评论列 {args.text_col!r} 和标签列 {args.label_col!r} 是否存在。"
        )

    print("\n2. 缺失值统计：")
    print(df[[args.text_col, args.label_col]].isnull().sum())

    print("\n3. 重复值数量：")
    print(df.duplicated().sum())

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[args.text_col, args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    configure_chinese_font()
    plt.figure(figsize=(6, 4))
    sns.countplot(x=args.label_col, data=df)
    plt.title("情感类别分布", fontsize=14)
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    label_plot_path = results_dir / "label_distribution.png"
    plt.savefig(label_plot_path, dpi=200)
    plt.close()
    print(f"\n4. 情感类别条形图已保存：{label_plot_path}")

    stopwords = load_stopwords(stopwords_path)
    content_seg = data_cleaning(df[args.text_col].astype(str).tolist(), stopwords)
    df["review_clean"] = content_seg

    print("\n5. 清洗后数据前 5 行：")
    print(df[[args.text_col, "review_clean", args.label_col]].head())

    print("\n6. 清洗后缺失值统计：")
    print(df[["review_clean", args.label_col]].isnull().sum())

    vectorizer = TfidfVectorizer(max_features=args.max_features)
    data_transform = vectorizer.fit_transform(df["review_clean"])
    x = data_transform.toarray()
    y = df[args.label_col].to_numpy()

    indices = df.index.to_numpy()
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x,
        y,
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n7. 高斯贝叶斯分类器评价指标：")
    print(f"准确率 accuracy：{accuracy:.4f}")
    print(f"精确率 precision：{precision:.4f}")
    print(f"召回率 recall：{recall:.4f}")
    print(f"F1 值 f1_score：{f1:.4f}")
    print(f"高斯贝叶斯分类器在测试集上的准确率为：{accuracy:.4f}")

    prediction_compare = pd.DataFrame(
        {
            "评论": df.loc[test_indices, args.text_col].to_numpy(),
            "真实值": y_test,
            "预测值": y_pred,
        }
    )
    compare_path = results_dir / "prediction_compare.csv"
    prediction_compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"\n8. 真实值和预测值对比已保存：{compare_path}")


if __name__ == "__main__":
    main()

