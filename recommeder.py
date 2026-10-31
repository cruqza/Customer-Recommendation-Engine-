from pathlib import Path
import csv


DATA_PATH = Path("data/purchases.csv")

def read_purchases(path:Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows). Decide what makes a purchase row unusable."""
    valid_rows = []
    rejected_rows = []

    #track unique rows to prevent duplicates
    seen_signatures = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            customer_id = str(row.get("customerid") or "").strip().upper()
            product_id = str(row.get("product_id") or "").strip().upper()
            # Check for missing or empty fields
            if not "customer_id" or not "product_id":
                row["rejection_reason"] = "missing or blank customer_id/product_id"
                rejected_rows.append(row)
                continue
            row["customer_id"] = customer_id
            row["product_id"] = product_id

            #validating the quantity rows for 0
            quantity_raw  = row.get("quantity", "0").strip()
            try:
                quantity = int(quantity_raw)
                if quantity <= 0:
                    raise ValueError 
            except ValueError:
                row["rejection_reason"] = f"Zero quantity, Invalid entry: {quantity_raw}"
                rejected_rows.append(row)
                continue 

            #create a unique footprint on sorted column items to catch identical rows
            row_unique = tuple(sorted(row.items()))
            if row_unique in seen_signatures:
                row["rejection_reason"] = "Duplicate row detected"
                rejected_rows.append(row)
                continue
            
            # If all checks pass, add to valid rows
            seen_signatures.add(row_unique)
            valid_rows.append(row)
    return valid_rows, rejected_rows


# Read the data and print the number of valid and rejected rows
valid, rejected = read_purchases(DATA_PATH)
print(f"Valid rows: {len(valid)}")
print(f"Rejected rows: {len(rejected)}")
