# Day 4 - Customer Recommendation Engine

Core-Python recommendation engine (no ML libraries). Uses only `csv`, `math`,
`collections` and `pathlib`.

## Engine flow

    data/purchases.csv -> read_purchases() -> RecommendationEngine
    -> recommend()/recommend_all() -> export_csv() -> output/recommendations.csv

Run from the project folder:

    python starter.py                                                # prints matrix, two customers, summary
    python -c "from day4_recommendation_engine import main; main()"
    # or the installed console script: day4-recommendation-engine

## 1. Read purchases (`read_purchases`)

Returns `(valid_rows, rejected_rows)`. A row is **rejected** when it fails any
of these rules:

| Rule | Rejection reason |
| --- | --- |
| missing/blank `customer_id`, `product_id`, or `product_name` | `missing or blank customer_id/product_id/product_name` |
| quantity not an integer, or `<= 0` | `zero or negative quantity: <raw>` |
| byte-identical duplicate footprint (same key/value set) | `duplicate row` |

Customer/product IDs are normalised to UPPER (`c106` -> `C106`), so only a
*truly* empty product makes a row unusable.

## 2. Histories

`RecommendationEngine.__init__` builds, **once** (later methods do not rescan
rows):

- `customer_to_products: dict[str, set]` -> product ids each customer bought
- `product_info: dict`                  -> product id to {name, category}
- `product_popularity: dict`            -> product id to purchase count
- `max_popularity`                      -> largest purchase count

`history(customer_id)` returns a customer's product set (or raises `KeyError`
for an unknown id).

## 3. Jaccard similarity

    jaccard(a, b) = |a intersect b| / |a union b|

Edge case: two customers with **both empty** sets are indistinguishable, so they
return `1.0`; an empty set against a non-empty set returns `0.0`.

## 4. Similar customers

`similar_customers(customer_id, k)` returns the top-`k` **other** customers,
best first, excluding the caller and anyone below `min_similarity`. Ties break
deterministically by ascending customer id.

## 5. Candidate products

`candidate_products(customer_id, k)` = the products that the customer's `k`
nearest neighbours bought that the customer **has not** bought, keyed by product
id. Each entry carries `supporters`, `similarity_sum`, `support_count` and
`popularity`. If no neighbour passed `min_similarity`, returns `{}` and
`recommend()` falls back to the most popular products the customer has not bought.

## 6. Ranking (`score`)

    score = similarity_sum * log2(1 + support_count) * (1 + popularity_factor)
    where popularity_factor = (popularity / max_popularity) ** 0.25

- `similarity_sum` -> how similar the supporting neighbours are (primary);
- `log2(1 + support_count)` -> consensus among neighbours;
- `(1 + popularity_factor)` -> mild popularity boost, capped at x2.

### Worked example

Customer C101 bought `P04` (Blue Band 500g) and `P08` (Ketepa Tea 250g). Its
top recommendation is **P03 (Festive Bread, Bakery)**. The neighbours who bought
P03 (and whose products C101 does not have) are {C117, C109, C113, C108}:

| neighbour | similarity to C101 |
| --- | --- |
| C117 | 0.5000 |
| C109 | 0.3333 |
| C113 | 0.1667 |
| C108 | 0.0000 |

similarity_sum = 1.0000, support_count = 4, popularity = 7 (= max_pop). Therefore

    score = 1.0 * log2(5) * (1 + (7/7)^0.25)
          = log2(5) * 2
          ~= 4.6439

## 7. RecommendationEngine

`RecommendationEngine(purchases).recommend_all(n=3)` returns one dict per
customer - every customer appears, even with empty recommendations. Unknown
customer ids raise `KeyError` - never nonsense.

## 8. Export CSV

`export_csv(rows, path)` writes `output/recommendations.csv` with columns:

    customer_id,product_id,product_name,category,score,support_count,supporters

One row per recommendation, plus one row per customer with no recommendations,
so every customer is present in the file.
