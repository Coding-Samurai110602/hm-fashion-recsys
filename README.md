# H&M Personalized Fashion Recommendations

## Description

A production-quality recommender system built on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle dataset. The task is to predict the 12 articles each customer will purchase in the next 7 days, evaluated by MAP@12.

The full project covers: (1) data engineering (CSV to Parquet), (2) exploratory data analysis, (3) candidate generation, (4) a learned ranking model, (5) offline evaluation (precision@k, recall@k, NDCG, MAP@12) with bootstrap confidence intervals, (6) simulated A/B testing with hypothesis testing, and (7) SHAP explainability.

**Data note:** Raw data is not committed to this repository (Kaggle competition terms). Download instructions are below.

## Dataset

| File | Rows | Description |
|------|------|-------------|
| articles.csv | 105,542 | Article metadata (product type, colour, department, etc.) |
| customers.csv | 1,371,980 | Customer demographics (age, club status, newsletter frequency) |
| transactions_train.csv | 31,788,324 | Purchase history (2018-09-20 to 2020-09-22) |
| sample_submission.csv | 1,371,980 | Submission format |

## Folder Structure

```
hm-fashion-recsys/
├── data/
│   ├── raw/          # original CSVs (read-only, not committed)
│   └── processed/    # parquet files (not committed)
├── notebooks/
│   └── 01_eda.ipynb  # full EDA with executed outputs
├── scripts/
│   ├── convert_to_parquet.py   # lossless CSV to Parquet conversion
│   └── verify_parquet.py       # 38-check verification suite
├── reports/
│   └── figures/      # 18 EDA plots (PNG)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# 1. Clone and create virtual environment
git clone <repo>
cd hm-fashion-recsys
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register Jupyter kernel
python -m ipykernel install --user --name hm-recsys --display-name "hm-recsys"

# 4. Authenticate with Kaggle (one-time)
kaggle auth login

# 5. Download only the CSV files (skips ~28 GB of images)
for f in articles.csv customers.csv transactions_train.csv sample_submission.csv; do
  kaggle competitions download \
    -c h-and-m-personalized-fashion-recommendations \
    -f "$f" -p data/raw
done
cd data/raw && for z in *.zip; do unzip -o "$z" && rm "$z"; done && cd ../..
chmod 444 data/raw/*.csv

# 6. Convert CSVs to Parquet
python scripts/convert_to_parquet.py

# 7. Verify conversion (all 38 checks must pass)
python scripts/verify_parquet.py

# 8. Open the EDA notebook
jupyter lab notebooks/01_eda.ipynb
```

## Key EDA Findings

| Finding | Verified Number |
|---------|----------------|
| Transaction date range | 2018-09-20 to 2020-09-22 |
| Week indexing | week_idx = (date - 2018-09-19).days // 7; weeks run Wed-Tue; target week = idx 104 |
| Exact duplicate transaction rows | 5,518,813 (17.4%); 91.8% have identical price (multi-unit purchases) |
| Customers who never purchased | 9,699 (0.7%) |
| Articles never sold | 995 (0.9%) |
| Zero-price transactions | 0 (minimum price = 0.000017) |
| Customer age range | 16-99; 9,553 customers under 18 |
| Article Gini coefficient | 0.759 |
| Top 10% articles share of purchases | 60.8% |
| Top 1% articles share of purchases | 18.6% |
| Median inter-purchase gap | 10 days (p25=4d, p75=24d) |
| Basket size p50 / p75 | 2 / 4 articles per customer-day |
| Customers active in only 1 week | 34.1% |
| Same-article repurchase rate (strict prior date) | 4.74% [95% CI: 4.72-4.76%] |
| Same product-code repurchase rate | 8.53% |
| Same product-type repurchase rate | 50.6% |
| Week-over-week top-12 article Jaccard | 0.200 (median 4 of 12 items carry over; 103 full-week pairs; 5 of 103 pairs J=0, dates consistent with seasonal transitions) |
| Week-over-week top-100 article Jaccard | 0.370 (median 54 of 100 items carry over) |
| Highest-volume weeks | Week 39 (2019-06-19): 635,891; Week 91 (2020-06-17): 549,443 |
| COVID-19 impact | Week 77 (2020-03-11): 205,868 tx, down 31.3% from week 76 (299,791); online share rose from 75% (wks 76-77) to 97-100% (wks 78-80) as stores closed; recovered by week 80 (2020-04-01) — note: sales_channel_id undocumented, channel 2 treated as online |
| Peak single days | 2019-09-28 (198,622 tx; 414 new-article debuts = 4.3× non-censored median of 96/day, rank #2 of 707 days, consistent with an autumn collection launch), 2020-04-11 Saturday before Easter (162,799), 2019-11-29 Black Friday (160,875) |
| Cold-start buyers in final week (0 prior) | 8.1% |
| New articles in final week | 3.7% (667/17,986) never sold before |
| Baseline A MAP@12 (global top-12) | 0.008748 |
| Baseline B MAP@12 (with deduplication, final) | 0.024457 (+179.6% vs A) |
| Baseline B MAP@12 without deduplication (ablation) | 0.023710 (delta: +0.000746 from dedup) |
| Customers with duplicate article_idx in raw top-12 history | 519,350 (38.3% of those with history; 47.6% of target-week buyers) |
| Day-of-week: Saturday lift vs Mon–Fri mean | 1.11× (48,090 avg vs 43,337 weekday mean) |

Both baseline results are deterministic: sort ties broken by article_idx ascending. The strong lift from Baseline B confirms that customer purchase history is the dominant signal. Any learned ranking model must clear MAP@12 = 0.0245 as the minimum bar.

**Popularity decay note:** For the top-500 articles by lifetime sales, median weekly sales remain above 50% of the debut-week level through 20+ weeks — these are survivorship-selected long sellers.

## Candidate Generation

Six candidate sources are evaluated on validation folds 100-103 (holdout week 104 reserved).

| Source | Recall@50 (mean±std) | k | Notes |
|--------|---------------------|---|-------|
| repurchase | 0.0352 ± 0.0018 | 50 | All history; +38% vs 12wk window |
| product_code | 0.0212 ± 0.0018 | 50 | 12wk lookback; variant-swapping |
| copurchase | 0.0248 ± 0.0019 | 50 | min_count=3 (best of {3,5,10}); ~44% customer coverage |
| popularity_last_week | 0.0969 ± 0.0101 | 200 | Recall@100; dominant single source |
| popularity_decayed | 0.0952 ± 0.0107 | 100 | Recall@100; inverse-time decay 1/(1+days) |
| segment_popular | 0.0558 ± 0.0066 | 50 | Per age-bucket; cold-start fallback |

**Merged recall@N (all sources, N=200 budget, mean ± std folds 100-103):**

| N | Recall | Std | Mean candidates per customer |
|---|--------|-----|------------------------------|
| 20 | 0.0377 | 0.0021 | — |
| 50 | 0.0595 | 0.0016 | — |
| 100 | 0.1018 | 0.0021 | — |
| 200 | 0.1750 | 0.0090 | 200 (eval, 100% at budget; k=200 final config) |

Coverage: 100% on all folds (popularity fills all customers). With `popularity_last_week k=200` (final config), all eval customers reach the full 200-candidate budget (mean=200, 100% at budget).

**Heuristic MAP@12 (priority ranking, production ordering):** 0.021849 ± 0.002307 across folds 100-103 (per fold: 0.021080, 0.019198, 0.022445, 0.024675).  
**Baseline B MAP@12 (EDA baseline):** 0.021411 ± 0.002208 — heuristic beats Baseline B on all 4 folds.  
**Oracle MAP@12:** 0.197 ± 0.012 (folds 100-103) — upper bound for a perfect ranker within the 200-candidate pool. 9.0× the heuristic; candidate precision is ~0.28% (~0.55 GT items per pool of 200 candidates; 0.28% × 200 ≈ 0.56).  
**Regression check (week 104):** recency-only ordering reproduces EDA Baseline B MAP@12 = 0.024457 exactly. Production ordering (last_date DESC, purchase_count DESC, article_idx ASC) gives MAP@12 = 0.024654 (+0.000198); 23,639 of 68,984 top-12 lists differ between the two orderings.  
**Scale test (k=200 final config):** 270.3M rows for 1.35M customers (all at budget=200); 166.2 MB parquet; 135s runtime; peak RSS 9,323 MB (measured via resource.getrusage).

Run the pipeline:
```bash
python scripts/run_candidates.py --folds 100 101 102 103 --n 20 50 100 200
python scripts/run_candidates.py --folds 103 --scale
python scripts/run_candidates.py --regression
```
