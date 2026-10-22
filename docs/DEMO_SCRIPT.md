# Demo Script — H&M Fashion Recommendation System

## Pre-recording setup

```bash
# 1. Start the stack (real bundle)
docker compose up
# 2. Open http://localhost:8080 in Chrome, browser zoom 100%
# 3. Close all other browser tabs
# 4. Clear localStorage: DevTools → Application → Local Storage → Clear All
# 5. Start screen recording software, set resolution 1440×900
```

---

## Section 1: Layman journey (2 min 30 sec)

**[Shot 1 — 15 sec]** Shop page (/) — full catalog visible, hero area in frame

"This is a fashion recommender built on 31 million H&M transactions. Let's create a profile and start shopping."

---

**[Shot 2 — 20 sec]** Create Profile modal — type a nickname, select an age range from the dropdown, click Save

"I enter a nickname and pick my age range — that's how the model tailors segment-based candidates to my demographic."

---

**[Shot 3 — 20 sec]** Browse catalog, click Add to Cart on three different articles

"I browse the catalog and add three items — these purchases become my shopping history."

---

**[Shot 4 — 15 sec]** Open cart drawer, review items, click Checkout

"Checkout saves my purchases and triggers the recommendation pipeline."

---

**[Shot 5 — 25 sec]** For You page (/for-you) — 12 recommendation cards with reason chips visible

"The model ranked 200 candidates with LightGBM LambdaRank and returned the top 12, each tagged with a plain-language reason — like 'trending 87% more than its 4-week average'."

---

**[Shot 6 — 20 sec]** Click "Why this?" on the first card, WhyDrawer opens showing SHAP feature rows

"The Why panel shows which SHAP features drove this recommendation — repurchase signal, age-group fit, or category affinity."

---

**[Shot 7 — 15 sec]** Click "Compare with simple rules" toggle, baseline column appears beside model column

"The baseline toggle shows what a popularity heuristic would recommend — our model beats it by +47.6% on the holdout week."

---

**[Shot 8 — 20 sec]** Navigate to How It Works page (/how-it-works), scroll through pipeline diagram and feature list

"How It Works explains the pipeline in plain language — six candidate sources, 68 features, a learned ranker — no ML background needed to follow along."

---

**[Shot 9 — 20 sec]** Navigate to Shoppers page (/shoppers), click "View picks" on one shopper card, hit-rate callouts appear

"The Shoppers page shows real H&M customers. I click 'View picks' to reveal how many of their actual purchases our ranker and the baseline both caught."

---

## Section 2: Engineering (60 sec)

**[Shot 10 — 15 sec]** Navigate to Engineering hub (/engineering), three sub-page cards visible

"Now the engineering view — three pages of technical evidence."

---

**[Shot 11 — 20 sec]** Model Lab (/engineering/lab) — scroll to ablation bar chart and SHAP attribution table

"The ablation study shows article features contribute the most MAP@12 lift, consistent with SHAP attribution — not gain importance, which over-weights high-cardinality categoricals."

---

**[Shot 12 — 15 sec]** A/B Simulator (/engineering/simulator) — sample-size output and false-positive peeking callout visible

"The A/B simulator uses the actual holdout hit-rate arrays: at the observed effect size, we need only 427 users per arm — and peeking inflates false-positive rate 4x."

---

**[Shot 13 — 10 sec]** System page (/engineering/system) — latency table in frame

"The System page shows measured latency: 56ms cold for recommendations, 1ms on cache hit."

---

## Section 3: GitHub (15 sec)

**[Shot 14 — 15 sec]** GitHub repository page — README visible, green CI badge in frame

"The full pipeline — candidate generation, feature engineering, ranker training, serving, and E2E tests — runs in GitHub Actions CI. 349 backend tests, 40 Playwright E2E tests, Docker build and smoke."

---

Total estimated time: 3 min 30 sec
