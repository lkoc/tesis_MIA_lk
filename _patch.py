import re

filepath = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\pinn_cables\io\readers.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find the load_domain function using regex
pattern = r'(def load_domain\(path: str \| Path\) -> Domain2D:\n.*?return Domain2D\(\*\*\{k: d\[k\] for k in \("xmin", "xmax", "ymin", "ymax"\)\}\))'
match = re.search(pattern, content, re.DOTALL)
if not match:
    print("ERROR: Could not find load_domain function")
    raise SystemExit(1)

old_func = match.group(1)
print("Found old function, length:", len(old_func))

new_func = '''def load_domain(path: str | Path) -> Domain2D:
    """Load domain extents from *domain.csv*.

    Accepted formats:

    * **Wide** (one row): columns ``xmin, xmax, ymin, ymax``.
    * **Long** (multiple rows): columns ``param, value`` with rows for
      ``xmin``, ``xmax``, ``ymin``, ``ymax``.
    """
    rows = _read_csv(path)
    d: dict[str, float] = {}
    if rows and "param" in rows[0] and "value" in rows[0]:
        # Long / key-value format
        for r in rows:
            d[r["param"].strip()] = float(r["value"])
    else:
        # Wide / columnar format - each column name is a parameter
        if len(rows) != 1:
            raise ValueError(
                f"Wide-format domain.csv must have exactly 1 data row, got {len(rows)}"
            )
        for k, v in rows[0].items():
            d[k.strip()] = float(v)
    required = {"xmin", "xmax", "ymin", "ymax"}
    missing = required - d.keys()
    if missing:
        raise ValueError(f"Missing domain parameters: {missing}")
    return Domain2D(**{k: d[k] for k in ("xmin", "xmax", "ymin", "ymax")})'''

content = content.replace(old_func, new_func)
with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("Patch applied successfully!")
