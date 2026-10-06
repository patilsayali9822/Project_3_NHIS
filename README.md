# Retail Large-Scale EDA Project

## Project
Exploratory Data Analysis of Large-Scale Retail Transaction Data

This project follows the requested EDA structure:
- Data inspection and quality checks
- Univariate analysis
- Outlier detection using IQR and Z-score
- Bivariate analysis
- Pearson correlation
- Multivariate analysis
- Skewness and kurtosis
- Business insights and exported reports

## Folder structure

```text
retail_eda_project/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── retail_large_dataset.csv   <-- PUT YOUR LARGE CSV HERE
│   └── README.txt
│
└── outputs/
    ├── figures/                   <-- generated automatically
    └── tables/                    <-- generated automatically
```

## 1. Put the dataset in the correct folder

Do NOT put the 100,000-row CSV inside the Python code.

Copy your file:

`retail_large_dataset.csv`

to:

`retail_eda_project/data/retail_large_dataset.csv`

The filename should be exactly `retail_large_dataset.csv`.

## 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install libraries

```bash
pip install -r requirements.txt
```

## 4. Run the project

From the `retail_eda_project` folder:

```bash
python main.py
```

## 5. Output

The program creates:

- `outputs/figures/` - histograms, boxplots, scatter plots, count plots, correlation heatmap and pair plot
- `outputs/tables/` - CSV statistical summaries and business analysis tables
- `outputs/eda_report.txt` - text report
- `outputs/eda_summary.xlsx` - Excel summary

The code calculates the actual findings from your CSV. It does not hard-code the example findings from the assignment.

## Expected dataset columns

The project expects these 18 columns:

`customer_id, age, gender, city, state, customer_segment, order_id, order_date, product_category, product_subcategory, product_price, quantity, discount_percentage, final_price, payment_method, shipping_type, delivery_days, return_status`

If your CSV uses different column names, update `EXPECTED_COLUMNS` and the related column names in `main.py`.

## Important

Do not upload the dataset to GitHub if it is confidential or too large. Keep it in the local `data/` folder and add it to `.gitignore`.
