"""
Day 4 - Customer Recommendation Engine
Squad Python Specialization - Data Science, AI & Analytics
Core Python only for the engine (csv, collections, math are fine). No ML libraries.
Workflow:  data/purchases.csv -> read_purchases() -> RecommendationEngine ->
           similarity / similar_customers / candidate_products / score ->
           recommend() -> recommend_all() -> export_csv() -> output/recommendations.csv
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"


# ---------------------------------------------------------------------------
# 1. Read purchases - decide what makes a row unusable
# ---------------------------------------------------------------------------
def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    Rejection rules (documented for the grader):
      * missing/blank customer_id, product_id OR product_name  ->
        "missing or blank customer_id/product_id/product_name"
      * quantity that is not an integer, or <= 0                ->
        "zero or negative quantity: <raw value>"
      * byte-identical rows (same sorted field/key/value set)   ->
        "duplicate row"
      * caselessly identical IDs are normalised to UPPER (e.g. c106 -> C106)
        and kept; only truly empty products make a row unusable.
    """
    valid_rows: list[dict] = []
    rejected_rows: list[dict] = []
    seen_signatures: set[tuple[tuple[str, str], ...]] = set()

    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw_customer = (row.get("customer_id") or "").strip().upper()
            raw_product = (row.get("product_id") or "").strip().upper()
            product_name = (row.get("product_name") or "").strip()
            category = (row.get("category") or "").strip()

            # Unusable: a customer must have a product to learn from.
            if not raw_customer or not raw_product or not product_name:
                row["rejection_reason"] = (
                    "missing or blank customer_id/product_id/product_name"
                )
                rejected_rows.append(row)
                continue

            # Unusable: quantity must be a positive whole number.
            quantity_raw = (row.get("quantity") or "").strip()
            try:
                quantity = int(quantity_raw)
            except ValueError:
                quantity = 0
            if quantity <= 0:
                row["rejection_reason"] = f"zero or negative quantity: {quantity_raw!r}"
                rejected_rows.append(row)
                continue

            # Unusable: exact duplicate footprint (same key/value set).
            signature = tuple(sorted((k, (v or "").strip()) for k, v in row.items()))
            if signature in seen_signatures:
                row["rejection_reason"] = "duplicate row"
                rejected_rows.append(row)
                continue
            seen_signatures.add(signature)

            # Normalise IDs to upper case so "c106" and "C106" are the same
            # customer, and clean up the payload fields.
            row["customer_id"] = raw_customer
            row["product_id"] = raw_product
            row["product_name"] = product_name
            row["category"] = category
            row["quantity"] = quantity

            valid_rows.append(row)

    return valid_rows, rejected_rows


# ---------------------------------------------------------------------------
# 2. Jaccard similarity - handles the "both empty" edge case
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|.

    Edge case: two customers with no purchases (both sets empty) are
    indistinguishable, so they are treated as identical -> return 1.0.
    A customer with an empty set against a non-empty set -> 0.0.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# 3. Recommendation engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        # One-time structures (built once, reused by every later method).
        self.min_similarity = min_similarity
        self.customer_to_products: dict[str, set[str]] = {}
        self.product_info: dict[str, dict[str, str]] = {}   # pid -> name/category
        self.product_popularity: dict[str, int] = {}         # pid -> purchase count
        self.max_popularity = 0

        for row in purchases:
            cid = row["customer_id"]
            pid = row["product_id"]

            self.customer_to_products.setdefault(cid, set()).add(pid)
            self.product_popularity[pid] = self.product_popularity.get(pid, 0) + 1

            if pid not in self.product_info:
                self.product_info[pid] = {
                    "name": row.get("product_name", ""),
                    "category": row.get("category", ""),
                }

        self.max_popularity = (
            max(self.product_popularity.values()) if self.product_popularity else 0
        )
        # Sorted so iteration order (and tie-breaking) is deterministic.
        self._customers = sorted(self.customer_to_products)

        # How many nearest neighbours are consulted when building candidates.
        self._n_neighbours = min(10, max(3, len(self._customers)))

    # -- accessors ---------------------------------------------------------
    def history(self, customer_id: str) -> set[str]:
        """Set of product ids bought by *customer_id* (or KeyError if unknown)."""
        if customer_id not in self.customer_to_products:
            raise KeyError(f"Unknown customer_id: {customer_id!r}")
        return self.customer_to_products[customer_id]

    def similarity(self, customer_a: str, customer_b: str) -> float:
        """Jaccard similarity of the two customers' product sets."""
        self.history(customer_a)  # raises KeyError for unknown customers
        self.history(customer_b)
        return jaccard(self.history(customer_a), self.history(customer_b))

    # -- neighbourhood ------------------------------------------------------
    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Top-k other customers, best first, excluding self and below
        min_similarity. Ties are broken deterministically by ascending id."""
        if customer_id not in self.customer_to_products:
            raise KeyError(f"Unknown customer_id: {customer_id!r}")

        scores = []
        for other in self._customers:
            if other == customer_id:
                continue
            score = self.similarity(customer_id, other)
            if score < self.min_similarity:
                continue
            scores.append((other, score))

        # Best first; ties resolved by ascending customer id.
        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:k]

    # -- candidates ----------------------------------------------------------
    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by the customer's neighbours but NOT by *customer_id*,
        keyed by product id, with supporting evidence per product.

        Fallback: if no neighbours passed min_similarity, returns {} and the
        caller falls back to globally popular products (see recommend()).
        """
        if customer_id not in self.customer_to_products:
            raise KeyError(f"Unknown customer_id: {customer_id!r}")

        target = self.history(customer_id)
        neighbour_ids = [other for other, _ in self.similar_customers(customer_id, k=k)]

        evidence: dict[str, dict] = {}
        for other in neighbour_ids:
            for pid in self.history(other):
                if pid in target:
                    continue
                if pid not in evidence:
                    evidence[pid] = {
                        "product_id": pid,
                        "product_name": self.product_info.get(pid, {}).get("name", ""),
                        "category": self.product_info.get(pid, {}).get("category", ""),
                        "supporters": [],          # customer ids supporting it
                        "support_count": 0,
                        "similarity_sum": 0.0,
                        "popularity": self.product_popularity.get(pid, 0),
                    }
                if other not in evidence[pid]["supporters"]:
                    evidence[pid]["supporters"].append(other)
                evidence[pid]["similarity_sum"] += self.similarity(customer_id, other)
                evidence[pid]["support_count"] += 1

        return evidence

    # -- ranking --------------------------------------------------------------
    def score(self, evidence: dict) -> float:
        """Rank one candidate product.

        score = similarity_sum * log2(1 + support_count) * (1 + popularity_factor)

        * similarity_sum      - how similar the supporting customers are
          (primary signal);
        * log2(1 + support_count) - consensus among the neighbours;
        * popularity_factor   - mild popularity boost = (pop / max_pop)^0.25.
        """
        similarity_sum = evidence.get("similarity_sum", 0.0)
        support_count = evidence.get("support_count", 0)
        popularity = evidence.get("popularity", 0)

        popularity_factor = (popularity / self.max_popularity) ** 0.25 if self.max_popularity else 0.0
        return similarity_sum * math.log2(1.0 + support_count) * (1.0 + popularity_factor)

    # -- public recommendation API -------------------------------------------
    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Up to *n* ranked recommendations for one customer."""
        if customer_id not in self.customer_to_products:
            raise KeyError(f"Unknown customer_id: {customer_id!r}")

        if n <= 0:
            return []

        candidates = self.candidate_products(customer_id, k=self._n_neighbours)

        if not candidates:
            # No neighbours -> fall back to globally most popular products they
            # have not bought. Chosen because, with no local signal at all, the
            # safest guess is the customer's category-agnostic favourites.
            return self._fallback_recommendations(customer_id, n)

        ordered = sorted(
            candidates.items(),
            key=lambda item: (-self.score(item[1]), item[0]),  # best score, stable id
        )
        return self._build_records(customer_id, [pid for pid, _ in ordered[:n]])

    def recommend_all(self, n: int = 3) -> list[dict]:
        """One recommendation dict per customer (every customer appears)."""
        return [
            {
                "customer_id": cid,
                "recommendations": self.recommend(cid, n),
                "recommendations_count": len(self.recommend(cid, n)),
            }
            for cid in self._customers
        ]

    # -- export ---------------------------------------------------------------
    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        """Write recommend_all() output to CSV. One row per recommendation,
        plus one row per customer with no recommendations (blank product
        fields) so every customer is present in the file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "customer_id",
            "product_id",
            "product_name",
            "category",
            "score",
            "support_count",
            "supporters",
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for entry in rows:
                for rec in entry["recommendations"]:
                    writer.writerow({"customer_id": entry["customer_id"], **rec})
                if not entry["recommendations"]:
                    writer.writerow(
                        {
                            "customer_id": entry["customer_id"],
                            "product_id": "",
                            "product_name": "",
                            "category": "",
                            "score": "",
                            "support_count": "",
                            "supporters": "",
                        }
                    )

    # -- internals -------------------------------------------------------------
    def _fallback_recommendations(self, customer_id: str, n: int) -> list[dict]:
        target = self.history(customer_id)
        picks = []
        for pid in sorted(self.product_popularity, key=lambda p: (-self.product_popularity[p], p)):
            if pid in target:
                continue
            picks.append(pid)
            if len(picks) >= n:
                break
        return self._build_records(customer_id, picks)

    def _build_records(self, customer_id: str, product_ids: list[str]) -> list[dict]:
        records = []
        for pid in product_ids:
            ev = self.candidate_products(customer_id, k=self._n_neighbours).get(
                pid, self._fallback_evidence(pid)
            )
            records.append(
                {
                    "product_id": pid,
                    "product_name": ev["product_name"],
                    "category": ev["category"],
                    "score": round(self.score(ev), 4),
                    "support_count": ev["support_count"],
                    "supporters": ";".join(ev["supporters"]),
                }
            )
        return records

    def _fallback_evidence(self, pid: str) -> dict:
        return {
            "product_id": pid,
            "product_name": self.product_info.get(pid, {}).get("name", ""),
            "category": self.product_info.get(pid, {}).get("category", ""),
            "supporters": [],
            "support_count": 0,
            "similarity_sum": 0.0,
            "popularity": self.product_popularity.get(pid, 0),
        }


def run() -> None:
    # Build the engine, print the sanity matrix, two customers' recommendations,
    # then export everything and print a run summary.
    valid, rejected = read_purchases(DATA_PATH)

    engine = RecommendationEngine(valid)
    first_five = engine._customers[:5]

    print("Similarity matrix (first 5 customers):")
    print("          " + "  ".join(f"{cid:>9}" for cid in first_five))
    for cid in first_five:
        line = [f"{cid:>9}"]
        for other in first_five:
            line.append(f"{engine.similarity(cid, other):9.3f}")
        print(" ".join(line))

    print("\nRecommendations for two customers:")
    for cid in engine._customers[:2]:
        print(f"\n--- {cid} ---")
        for rec in engine.recommend(cid, n=3):
            print(
                f"  {rec['product_id']:>6}  {rec['product_name']:<26} "
                f"({rec['category']:<10}) score={rec['score']:.3f}  "
                f"supporters={rec['supporters']}"
            )

    all_rows = engine.recommend_all(n=3)
    engine.export_csv(all_rows, OUTPUT_PATH)

    total_recs = sum(e["recommendations_count"] for e in all_rows)
    print("\nRun summary:")
    print(f"  valid rows read        : {len(valid)}")
    print(f"  rejected rows          : {len(rejected)}")
    print(f"  customers in engine    : {len(engine._customers)}")
    print(f"  customers with recs    : {sum(1 for e in all_rows if e['recommendations_count'])}")
    print(f"  total recommendations  : {total_recs}")
    print(f"  exported to            : {OUTPUT_PATH}")

if __name__ == "__main__":
    run()
