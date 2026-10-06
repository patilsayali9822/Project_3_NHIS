"""
Exploratory Data Analysis - Large-Scale Retail Transaction Dataset
Project: NextHikes IT Solutions

Expected input:
    data/retail_large_dataset.csv

Expected columns (18):
customer_id, age, gender, city, state, customer_segment, order_id,
order_date, product_category, product_subcategory, product_price,
quantity, discount_percentage, final_price, payment_method,
shipping_type, delivery_days, return_status

Run:
    python main.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "retail_large_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

EXPECTED_COLUMNS = [
    "customer_id", "age", "gender", "city", "state", "customer_segment",
    "order_id", "order_date", "product_category", "product_subcategory",
    "product_price", "quantity", "discount_percentage", "final_price",
    "payment_method", "shipping_type", "delivery_days", "return_status"
]

NUMERIC_COLUMNS = [
    "age", "product_price", "quantity", "discount_percentage",
    "final_price", "delivery_days"
]

CATEGORICAL_COLUMNS = [
    "gender", "city", "state", "customer_segment", "product_category",
    "product_subcategory", "payment_method", "shipping_type", "return_status"
]


def setup_folders():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n\n"
            "Put your retail_large_dataset.csv file inside the data folder."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Standardize column names and string whitespace.
    df.columns = df.columns.str.strip().str.lower()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()

    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected:
        raise ValueError(
            "These expected columns are missing from the CSV:\n"
            + ", ".join(missing_expected)
            + "\n\nColumns found:\n"
            + ", ".join(df.columns)
        )

    # Convert date/numeric fields safely.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_table(df, filename):
    df.to_csv(TABLE_DIR / filename, index=True)


def univariate_analysis(df):
    print("\n=== UNIVARIATE ANALYSIS ===")

    numeric_summary = df[NUMERIC_COLUMNS].describe().T
    numeric_summary["median"] = df[NUMERIC_COLUMNS].median()
    numeric_summary["skewness"] = df[NUMERIC_COLUMNS].skew()
    numeric_summary["kurtosis"] = df[NUMERIC_COLUMNS].kurtosis()
    numeric_summary["missing_values"] = df[NUMERIC_COLUMNS].isna().sum()
    numeric_summary.to_csv(TABLE_DIR / "numeric_summary.csv")

    categorical_summary = {}
    for col in CATEGORICAL_COLUMNS:
        counts = df[col].value_counts(dropna=False)
        counts.name = "count"
        counts.to_csv(TABLE_DIR / f"{col}_frequency.csv")
        categorical_summary[col] = counts

    print(numeric_summary.round(3))

    # Histograms for all numerical columns.
    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(9, 5))
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hist_{col}.png", dpi=150)
        plt.close()

    # Boxplots for numerical columns.
    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(9, 4))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"box_{col}.png", dpi=150)
        plt.close()

    # Important categorical count plots.
    for col in ["product_category", "customer_segment", "payment_method", "return_status"]:
        plt.figure(figsize=(9, 5))
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order)
        plt.title(f"Count by {col}")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"count_{col}.png", dpi=150)
        plt.close()


def outlier_analysis(df):
    print("\n=== OUTLIER ANALYSIS ===")

    rows = []
    for col in NUMERIC_COLUMNS:
        s = df[col].dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_count = int(((s < lower) | (s > upper)).sum())

        mean = s.mean()
        std = s.std()
        if std == 0 or pd.isna(std):
            z_count = 0
        else:
            z_scores = ((s - mean) / std).abs()
            z_count = int((z_scores > 3).sum())

        rows.append({
            "column": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "IQR_outliers": iqr_count,
            "Zscore_gt_3_outliers": z_count
        })

    outlier_df = pd.DataFrame(rows)
    outlier_df.to_csv(TABLE_DIR / "outlier_summary.csv", index=False)
    print(outlier_df.round(3))


def bivariate_analysis(df):
    print("\n=== BIVARIATE ANALYSIS ===")

    numeric = df[NUMERIC_COLUMNS].copy()
    corr = numeric.corr(method="pearson")
    corr.to_csv(TABLE_DIR / "pearson_correlation.csv")

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    # Key numerical relationships.
    pairs = [
        ("product_price", "final_price"),
        ("quantity", "final_price"),
        ("discount_percentage", "final_price"),
        ("age", "final_price"),
        ("delivery_days", "final_price"),
    ]

    for x, y in pairs:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.35, s=25)
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"scatter_{x}_vs_{y}.png", dpi=150)
        plt.close()

    # Numerical vs categorical analyses.
    segment_spending = (
        df.groupby("customer_segment", dropna=False)["final_price"]
        .agg(["count", "mean", "median", "sum"])
        .sort_values("mean", ascending=False)
    )
    segment_spending.to_csv(TABLE_DIR / "customer_segment_vs_spending.csv")

    category_return = (
        df.assign(
            return_flag=df["return_status"].astype("string").str.lower().eq("yes").astype(int)
        )
        .groupby("product_category", dropna=False)["return_flag"]
        .agg(["count", "mean"])
        .rename(columns={"mean": "return_rate"})
        .sort_values("return_rate", ascending=False)
    )
    category_return["return_rate_percent"] = category_return["return_rate"] * 100
    category_return.to_csv(TABLE_DIR / "category_vs_return_rate.csv")

    shipping_delivery = (
        df.groupby("shipping_type", dropna=False)["delivery_days"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("mean")
    )
    shipping_delivery.to_csv(TABLE_DIR / "shipping_vs_delivery_days.csv")

    # Visualizations.
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=segment_spending.reset_index(),
        x="customer_segment",
        y="mean",
        errorbar=None
    )
    plt.title("Average Final Price by Customer Segment")
    plt.ylabel("Average Final Price")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "segment_vs_average_final_price.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    temp = category_return.reset_index()
    sns.barplot(data=temp, x="product_category", y="return_rate_percent", errorbar=None)
    plt.title("Return Rate by Product Category")
    plt.ylabel("Return Rate (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "category_vs_return_rate.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    temp = shipping_delivery.reset_index()
    sns.barplot(data=temp, x="shipping_type", y="mean", errorbar=None)
    plt.title("Average Delivery Days by Shipping Type")
    plt.ylabel("Average Delivery Days")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shipping_vs_delivery_days.png", dpi=150)
    plt.close()

    return corr


def multivariate_analysis(df):
    print("\n=== MULTIVARIATE ANALYSIS ===")

    # Pair plot on a sample so the project remains practical for 100k rows.
    pair_cols = [
        "product_price", "quantity", "discount_percentage",
        "final_price", "delivery_days"
    ]
    sample_n = min(3000, len(df))
    sample = df[pair_cols].dropna().sample(sample_n, random_state=42)

    g = sns.pairplot(sample, corner=True, plot_kws={"alpha": 0.25, "s": 18})
    g.fig.suptitle("Multivariate Pair Plot", y=1.02)
    g.fig.savefig(FIG_DIR / "multivariate_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Business interaction tables.
    category_discount_price = (
        df.groupby("product_category", dropna=False)
        .agg(
            avg_discount=("discount_percentage", "mean"),
            avg_final_price=("final_price", "mean"),
            total_revenue=("final_price", "sum"),
            transactions=("order_id", "count"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    category_discount_price.to_csv(TABLE_DIR / "category_discount_final_price.csv")

    segment_age_spending = (
        df.assign(age_group=pd.cut(
            df["age"],
            bins=[0, 29, 44, 59, np.inf],
            labels=["Under 30", "30-45", "46-60", "61+"],
            include_lowest=True
        ))
        .groupby(["customer_segment", "age_group"], observed=False)["final_price"]
        .agg(["count", "mean", "median", "sum"])
        .sort_values("mean", ascending=False)
    )
    segment_age_spending.to_csv(TABLE_DIR / "segment_age_spending.csv")

    shipping_return = (
        df.assign(
            return_flag=df["return_status"].astype("string").str.lower().eq("yes").astype(int)
        )
        .groupby("shipping_type", dropna=False)
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            return_rate=("return_flag", "mean"),
            transactions=("order_id", "count"),
        )
        .sort_values("return_rate", ascending=False)
    )
    shipping_return["return_rate_percent"] = shipping_return["return_rate"] * 100
    shipping_return.to_csv(TABLE_DIR / "shipping_delivery_return.csv")


def generate_report(df, corr):
    total_revenue = df["final_price"].sum()
    return_rate = (
        df["return_status"].astype("string").str.lower().eq("yes").mean() * 100
    )

    top_category = df["product_category"].value_counts().idxmax()
    top_segment = df["customer_segment"].value_counts().idxmax()

    corr_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    corr_pairs = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
    strongest = corr_pairs[:5]

    report = []
    report.append("EXPLORATORY DATA ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"Rows: {len(df):,}")
    report.append(f"Columns: {len(df.columns)}")
    report.append(f"Total final_price / revenue: {total_revenue:,.2f}")
    report.append(f"Overall return rate: {return_rate:.2f}%")
    report.append(f"Most frequent product category: {top_category}")
    report.append(f"Most frequent customer segment: {top_segment}")
    report.append("")
    report.append("Missing values")
    report.append("-" * 70)
    missing = df.isna().sum().sort_values(ascending=False)
    for col, count in missing.items():
        report.append(f"{col}: {int(count):,}")

    report.append("")
    report.append("Strongest Pearson correlations")
    report.append("-" * 70)
    for a, b, value in strongest:
        report.append(f"{a} vs {b}: {value:.4f}")

    report.append("")
    report.append("Important note")
    report.append("-" * 70)
    report.append(
        "Outliers were identified using IQR and Z-score methods. "
        "They are reported but not automatically deleted because high-value "
        "transactions may be genuine business observations."
    )

    (OUTPUT_DIR / "eda_report.txt").write_text("\n".join(report), encoding="utf-8")


def save_excel_report(df, corr):
    excel_file = OUTPUT_DIR / "eda_summary.xlsx"

    numeric_summary = df[NUMERIC_COLUMNS].describe().T
    numeric_summary["median"] = df[NUMERIC_COLUMNS].median()
    numeric_summary["skewness"] = df[NUMERIC_COLUMNS].skew()
    numeric_summary["kurtosis"] = df[NUMERIC_COLUMNS].kurtosis()

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        numeric_summary.to_excel(writer, sheet_name="Numeric Summary")
        corr.to_excel(writer, sheet_name="Correlation")
        df.isna().sum().rename("missing_values").to_excel(
            writer, sheet_name="Missing Values"
        )
        df["product_category"].value_counts().rename("count").to_excel(
            writer, sheet_name="Category Counts"
        )
        df["customer_segment"].value_counts().rename("count").to_excel(
            writer, sheet_name="Segment Counts"
        )
        df["return_status"].value_counts().rename("count").to_excel(
            writer, sheet_name="Return Status"
        )


def main():
    setup_folders()
    df = load_data()

    print(f"Dataset loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Basic inspection.
    df.head(10).to_csv(TABLE_DIR / "first_10_rows.csv", index=False)
    df.dtypes.astype(str).rename("dtype").to_csv(TABLE_DIR / "data_types.csv")
    df.isna().sum().rename("missing_values").to_csv(TABLE_DIR / "missing_values.csv")

    duplicates = int(df.duplicated().sum())
    (TABLE_DIR / "data_quality_summary.txt").write_text(
        f"Rows: {len(df):,}\n"
        f"Columns: {len(df.columns)}\n"
        f"Duplicate rows: {duplicates:,}\n",
        encoding="utf-8"
    )

    univariate_analysis(df)
    outlier_analysis(df)
    corr = bivariate_analysis(df)
    multivariate_analysis(df)
    generate_report(df, corr)
    save_excel_report(df, corr)

    print("\nEDA completed successfully.")
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TABLE_DIR}")
    print(f"Report:  {OUTPUT_DIR / 'eda_report.txt'}")
    print(f"Excel:   {OUTPUT_DIR / 'eda_summary.xlsx'}")


if __name__ == "__main__":
    main()
