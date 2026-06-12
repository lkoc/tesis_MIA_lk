"""Refresh zotero_network.html from the active Zotero database on Windows.

The database is opened read-only. The script uses the Tesis_MIA_2026
collection tree, Zotero tags, notes, and local PDF attachments as its source
of truth while preserving the existing HTML interface.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


ROOT_COLLECTION = "Tesis_MIA_2026"

THEMES = {
    "tema-pinn": {
        "label": "PINNs e IA física",
        "color": "#7b1fa2",
        "order": 1,
        "collections": {"01 tema-pinn - PINNs"},
    },
    "tema-transferencia-calor": {
        "label": "Transferencia de calor",
        "color": "#b71c1c",
        "order": 2,
        "collections": {"02 tema-transferencia-calor"},
    },
    "tema-suelo-backfill-humedad": {
        "label": "Suelo, backfill y humedad",
        "color": "#1b5e20",
        "order": 3,
        "collections": {"03 tema-suelo-backfill-humedad"},
    },
    "tema-fem-numerico": {
        "label": "FEM y métodos numéricos",
        "color": "#bf360c",
        "order": 4,
        "collections": {"04 tema-fem-numerico"},
    },
    "tema-cables-ampacidad": {
        "label": "Cables y ampacidad",
        "color": "#e65100",
        "order": 5,
        "collections": {"05 tema-cables-ampacidad"},
    },
    "tema-dtr-cargas-ciclicas": {
        "label": "DTR y cargas cíclicas",
        "color": "#0d47a1",
        "order": 6,
        "collections": {"06 tema-dtr-cargas-ciclicas"},
    },
    "tema-normas-estandares": {
        "label": "Normas y estándares",
        "color": "#37474f",
        "order": 7,
        "collections": {"07 tema-normas-estandares"},
    },
    "tema-otros": {
        "label": "Otros",
        "color": "#757575",
        "order": 8,
        "collections": set(),
    },
}

PAPER_TYPES = {
    "estado_arte": {"label": "Estado del arte", "color": "#00838f"},
    "aporte": {"label": "Aporte teórico / metodológico", "color": "#6a1b9a"},
    "aplicacion": {"label": "Aplicación", "color": "#e65100"},
    "normativa": {"label": "Normativa / Estándar", "color": "#37474f"},
    "fuente_clasica": {"label": "Fuente clásica", "color": "#4e342e"},
    "apoyo": {"label": "Apoyo / Didáctico", "color": "#558b2f"},
    "guia_tecnica": {"label": "Guía técnica", "color": "#0277bd"},
}

COMMUNITY_PALETTE = [
    "#c62828",
    "#6a1b9a",
    "#1565c0",
    "#2e7d32",
    "#e65100",
    "#00695c",
    "#827717",
]

STOP_WORDS = {
    "the", "and", "for", "with", "from", "into", "this", "that", "these",
    "those", "using", "used", "based", "study", "analysis", "method",
    "methods", "model", "models", "results", "result", "approach", "paper",
    "system", "systems", "application", "applications", "new", "power",
    "thermal", "heat", "cable", "cables", "underground", "temperature",
    "transfer", "current", "rating", "de", "del", "las", "los", "para",
    "por", "con", "una", "uno", "que", "como", "este", "esta", "sobre",
    "entre", "nosource", "reference", "tipo", "tema", "tesis", "mia",
}

META_TAG_PREFIXES = (
    "tema-",
    "tipo-",
    "tesis-mia-",
    "coleccion-",
)


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def norm_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = value.rstrip(" .")
    return "" if value in {"", "n/a", "na", "none"} else value


def extract_year(value: str) -> str:
    match = re.search(r"\b(18|19|20|21)\d{2}\b", value or "")
    return match.group(0) if match else "0000"


def windows_zotero_candidates() -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("ZOTERO_DATA_DIR"):
        candidates.append(Path(os.environ["ZOTERO_DATA_DIR"]) / "zotero.sqlite")

    home = Path.home()
    candidates.extend(
        [
            home / "Zotero" / "zotero.sqlite",
            Path(os.environ.get("APPDATA", "")) / "Zotero" / "zotero.sqlite",
        ]
    )

    profiles = Path(os.environ.get("APPDATA", "")) / "Zotero" / "Zotero" / "Profiles"
    if profiles.exists():
        for prefs in profiles.glob("*/prefs.js"):
            try:
                text = prefs.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            match = re.search(
                r'user_pref\("extensions\.zotero\.dataDir",\s*"([^"]+)"\)',
                text,
            )
            if match:
                raw = match.group(1).replace("\\\\", "\\")
                candidates.append(Path(raw) / "zotero.sqlite")
            candidates.append(prefs.parent / "zotero.sqlite")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def locate_zotero_db(explicit: str | None) -> Path:
    if explicit:
        db = Path(explicit).expanduser().resolve()
        if not db.exists():
            raise FileNotFoundError(f"No existe la base Zotero: {db}")
        return db

    existing = [path for path in windows_zotero_candidates() if path.exists()]
    if not existing:
        checked = "\n".join(f"  - {path}" for path in windows_zotero_candidates())
        raise FileNotFoundError(f"No se encontro zotero.sqlite. Rutas revisadas:\n{checked}")
    return max(existing, key=lambda path: path.stat().st_mtime)


def field_values(cur: sqlite3.Cursor, item_ids: list[int]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = defaultdict(dict)
    placeholders = ",".join("?" for _ in item_ids)
    query = f"""
        SELECT d.itemID, f.fieldName, v.value
        FROM itemData d
        JOIN fields f ON f.fieldID = d.fieldID
        JOIN itemDataValues v ON v.valueID = d.valueID
        WHERE d.itemID IN ({placeholders})
    """
    for item_id, field_name, value in cur.execute(query, item_ids):
        result[item_id][field_name] = value
    return result


def collection_tree_items(
    cur: sqlite3.Cursor,
    root_collection: str,
) -> tuple[int, list[int]]:
    root = cur.execute(
        """
        SELECT collectionID, libraryID
        FROM collections
        WHERE parentCollectionID IS NULL AND collectionName = ?
        ORDER BY collectionID
        LIMIT 1
        """,
        (root_collection,),
    ).fetchone()
    if not root:
        raise RuntimeError(f"No existe la coleccion raiz {root_collection!r}")

    rows = cur.execute(
        """
        WITH RECURSIVE tree(collectionID) AS (
            SELECT collectionID FROM collections WHERE collectionID = ?
            UNION ALL
            SELECT c.collectionID
            FROM collections c
            JOIN tree t ON c.parentCollectionID = t.collectionID
        )
        SELECT DISTINCT i.itemID
        FROM collectionItems ci
        JOIN tree t ON t.collectionID = ci.collectionID
        JOIN items i ON i.itemID = ci.itemID
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
          AND i.itemID NOT IN (SELECT itemID FROM itemAttachments)
          AND i.itemID NOT IN (SELECT itemID FROM itemNotes)
          AND it.typeName NOT IN ('attachment', 'note', 'annotation')
        ORDER BY i.itemID
        """,
        (root["collectionID"],),
    ).fetchall()
    return root["libraryID"], [row["itemID"] for row in rows]


def extract_note_text(note_html: str, title: str) -> tuple[int, str]:
    text = html.unescape(re.sub(r"<[^>]+>", " ", note_html or ""))
    text = " ".join(text.split())
    title_norm = norm_text(title)

    if "aporte a la investigacion" in title_norm:
        priority = 3
        pattern = r"Aporte a la investigaci[oó]n:\s*(.*?)(?=Clasificaci[oó]n depurada:|$)"
    elif "aporte a la tesis" in title_norm:
        priority = 2
        pattern = r"Aporte a la tesis:\s*(.*?)(?=Clasificaci[oó]n:|$)"
    else:
        priority = 1
        pattern = r"APORTE PARA LA TESIS:\s*(.*?)(?=LIMITACIONES:|$)"

    match = re.search(pattern, text, flags=re.I)
    return priority, (match.group(1).strip() if match else "")


def attachment_uri(storage: Path, attachment_key: str, stored_path: str) -> str:
    if not stored_path:
        return ""
    if stored_path.startswith("storage:"):
        path = storage / attachment_key / stored_path[8:]
    elif stored_path.startswith("file://"):
        return stored_path
    else:
        path = Path(stored_path)
    return path.resolve().as_uri() if path.exists() else ""


def extract_records(
    db: Path,
    root_collection: str,
) -> tuple[list[dict], dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    library_id, item_ids = collection_tree_items(cur, root_collection)
    if not item_ids:
        con.close()
        return [], {}

    placeholders = ",".join("?" for _ in item_ids)
    values = field_values(cur, item_ids)
    records: dict[int, dict] = {}
    for row in cur.execute(
        f"""
        SELECT i.itemID, i.key, it.typeName, i.dateAdded, i.dateModified
        FROM items i
        JOIN itemTypes it ON it.itemTypeID = i.itemTypeID
        WHERE i.itemID IN ({placeholders})
        ORDER BY i.itemID
        """,
        item_ids,
    ):
        item_id = row["itemID"]
        records[item_id] = {
            "item_id": item_id,
            "item_key": row["key"],
            "item_type": row["typeName"],
            "date_added": row["dateAdded"] or "",
            "date_modified": row["dateModified"] or "",
            "fields": values.get(item_id, {}),
            "creators": [],
            "tags": [],
            "collections": [],
            "attachments": [],
            "notes": [],
        }

    for row in cur.execute(
        f"""
        SELECT ic.itemID, c.firstName, c.lastName, c.fieldMode,
               ct.creatorType, ic.orderIndex
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
        WHERE ic.itemID IN ({placeholders})
        ORDER BY ic.itemID, ic.orderIndex
        """,
        item_ids,
    ):
        name = " ".join(part for part in (row["firstName"], row["lastName"]) if part).strip()
        if name:
            records[row["itemID"]]["creators"].append((row["creatorType"], name))

    for row in cur.execute(
        f"""
        SELECT it.itemID, t.name
        FROM itemTags it
        JOIN tags t ON t.tagID = it.tagID
        WHERE it.itemID IN ({placeholders})
        ORDER BY it.itemID, lower(t.name)
        """,
        item_ids,
    ):
        records[row["itemID"]]["tags"].append(row["name"])

    for row in cur.execute(
        f"""
        SELECT ci.itemID, c.collectionName
        FROM collectionItems ci
        JOIN collections c ON c.collectionID = ci.collectionID
        WHERE ci.itemID IN ({placeholders})
        ORDER BY ci.itemID, c.collectionName
        """,
        item_ids,
    ):
        records[row["itemID"]]["collections"].append(row["collectionName"])

    storage = db.parent / "storage"
    for row in cur.execute(
        f"""
        SELECT ia.parentItemID, ia.path, ia.contentType, i.key
        FROM itemAttachments ia
        JOIN items i ON i.itemID = ia.itemID
        WHERE ia.parentItemID IN ({placeholders})
          AND lower(COALESCE(ia.contentType, '')) = 'application/pdf'
        ORDER BY ia.parentItemID, i.itemID
        """,
        item_ids,
    ):
        records[row["parentItemID"]]["attachments"].append(
            {
                "key": row["key"],
                "path": row["path"] or "",
                "local_uri": attachment_uri(storage, row["key"], row["path"] or ""),
            }
        )

    for row in cur.execute(
        f"""
        SELECT n.parentItemID, n.title, n.note
        FROM itemNotes n
        WHERE n.parentItemID IN ({placeholders})
        ORDER BY n.parentItemID, n.itemID
        """,
        item_ids,
    ):
        priority, text = extract_note_text(row["note"] or "", row["title"] or "")
        if text:
            records[row["parentItemID"]]["notes"].append((priority, text))

    group = cur.execute(
        "SELECT groupID, name FROM groups WHERE libraryID = ? LIMIT 1",
        (library_id,),
    ).fetchone()
    metadata = {
        "db": str(db),
        "storage": str(storage),
        "library_id": library_id,
        "group_id": group["groupID"] if group else None,
        "group_name": group["name"] if group else "",
        "db_modified": db.stat().st_mtime,
        "raw_items": len(records),
    }
    con.close()
    return list(records.values()), metadata


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def merge_unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            result.append(value)
            seen.add(key)
    return result


def deduplicate(records: list[dict]) -> list[dict]:
    union = UnionFind(len(records))
    by_doi: dict[str, int] = {}
    by_title_year: dict[str, int] = {}

    for index, record in enumerate(records):
        fields = record["fields"]
        doi = norm_doi(fields.get("DOI", ""))
        title = norm_text(fields.get("title", ""))
        year = extract_year(fields.get("date", ""))
        title_key = f"{title}|{year}" if title else ""

        if doi and doi in by_doi:
            union.union(index, by_doi[doi])
        elif doi:
            by_doi[doi] = index
        if title_key and title_key in by_title_year:
            union.union(index, by_title_year[title_key])
        elif title_key:
            by_title_year[title_key] = index

    groups: dict[int, list[dict]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[union.find(index)].append(record)

    merged: list[dict] = []
    for members in groups.values():
        def richness(record: dict) -> tuple[int, int, int, int]:
            fields = record["fields"]
            authors = [name for kind, name in record["creators"] if kind == "author"]
            return (
                int(any(att["local_uri"] for att in record["attachments"])),
                len(fields.get("abstractNote", "")),
                len(authors),
                len(record["tags"]),
            )

        primary = max(members, key=richness)
        field_names = set().union(*(member["fields"].keys() for member in members))
        fields: dict[str, str] = {}
        for field_name in field_names:
            candidates = [member["fields"].get(field_name, "") for member in members]
            fields[field_name] = max(candidates, key=len, default="")

        creators: list[tuple[str, str]] = []
        for member in members:
            creators.extend(member["creators"])
        creators = [
            (kind, name)
            for kind, name in dict.fromkeys(creators)
        ]

        notes = [note for member in members for note in member["notes"]]
        best_note = max(notes, key=lambda note: (note[0], len(note[1])), default=(0, ""))
        attachments = [
            attachment
            for member in members
            for attachment in member["attachments"]
        ]
        attachments.sort(key=lambda att: (not bool(att["local_uri"]), att["key"]))

        merged.append(
            {
                "item_id": min(member["item_id"] for member in members),
                "item_ids": sorted(member["item_id"] for member in members),
                "item_key": primary["item_key"],
                "item_keys": merge_unique([member["item_key"] for member in members]),
                "item_type": primary["item_type"],
                "fields": fields,
                "creators": creators,
                "tags": merge_unique(
                    [tag for member in members for tag in member["tags"]]
                ),
                "collections": merge_unique(
                    [
                        collection
                        for member in members
                        for collection in member["collections"]
                    ]
                ),
                "attachments": attachments,
                "aporte": best_note[1],
            }
        )

    return sorted(
        merged,
        key=lambda record: (
            -int(extract_year(record["fields"].get("date", "")) or 0),
            norm_text(record["fields"].get("title", "")),
        ),
    )


def infer_themes(record: dict) -> list[str]:
    tag_keys = {norm_text(tag).replace(" ", "-") for tag in record["tags"]}
    collections = set(record["collections"])
    themes = []
    for key, info in THEMES.items():
        if key == "tema-otros":
            continue
        if key in tag_keys or collections.intersection(info["collections"]):
            themes.append(key)

    if themes:
        return themes

    fields = record["fields"]
    haystack = norm_text(
        " ".join(
            [
                fields.get("title", ""),
                fields.get("abstractNote", ""),
                " ".join(record["tags"]),
            ]
        )
    )
    rules = [
        ("tema-pinn", ("physics informed", "pinn", "scientific machine learning")),
        ("tema-dtr-cargas-ciclicas", ("dynamic thermal", "cyclic", "emergency rating")),
        ("tema-suelo-backfill-humedad", ("soil", "backfill", "bedding", "moisture")),
        ("tema-fem-numerico", ("finite element", "finite volume", "finite difference", "fem")),
        ("tema-cables-ampacidad", ("ampacity", "current rating", "power cable")),
        ("tema-normas-estandares", ("iec ", "ieee ", "astm ", "cigre ")),
        ("tema-transferencia-calor", ("heat conduction", "heat transfer", "diffusion equation")),
    ]
    inferred = [theme for theme, needles in rules if any(needle in haystack for needle in needles)]
    return inferred or ["tema-otros"]


def paper_type(record: dict) -> str:
    tags = [norm_text(tag).replace(" ", "-") for tag in record["tags"]]
    title = norm_text(record["fields"].get("title", ""))
    item_type = record["item_type"]

    if any(tag.startswith("tipo-estado-del-arte") or tag == "tipo-systematic-review" for tag in tags):
        return "estado_arte"
    if any(tag.startswith("tipo-aplicacion") for tag in tags):
        return "aplicacion"
    if any(tag.startswith("tipo-guia-tecnica") or tag.startswith("tipo-catalogo") for tag in tags):
        return "guia_tecnica"
    if any(tag.startswith("tipo-apoyo") or tag.startswith("tipo-aporte-didactico") for tag in tags):
        return "apoyo"
    if any(tag.startswith("tipo-fuente-clasica") for tag in tags):
        return "fuente_clasica"
    if any(tag.startswith("tipo-normativa") for tag in tags):
        return "normativa"
    if item_type == "document" and re.search(r"\b(iec|ieee|astm|standard)\b", title):
        return "normativa"
    return "aporte"


def primary_theme(themes: list[str], kind: str) -> str:
    if kind == "normativa" and "tema-normas-estandares" in themes:
        return "tema-normas-estandares"
    priority = [
        "tema-pinn",
        "tema-dtr-cargas-ciclicas",
        "tema-suelo-backfill-humedad",
        "tema-fem-numerico",
        "tema-cables-ampacidad",
        "tema-transferencia-calor",
        "tema-normas-estandares",
        "tema-otros",
    ]
    return next((theme for theme in priority if theme in themes), "tema-otros")


def tokenize(text: str) -> list[str]:
    normalized = norm_text(text)
    return [
        word
        for word in normalized.split()
        if len(word) >= 3 and word not in STOP_WORDS and not word.isdigit()
    ]


def semantic_text(record: dict) -> str:
    fields = record["fields"]
    useful_tags = [
        tag
        for tag in record["tags"]
        if not norm_text(tag).replace(" ", "-").startswith(META_TAG_PREFIXES)
        and norm_text(tag) not in {"nosource", "reference"}
    ]
    publication = fields.get("publicationTitle") or fields.get("bookTitle") or ""
    return " ".join(
        [
            (fields.get("title", "") + " ") * 3,
            (" ".join(useful_tags) + " ") * 2,
            fields.get("abstractNote", ""),
            publication,
        ]
    )


def tfidf_vectors(records: list[dict]) -> tuple[list[dict[str, float]], list[list[str]]]:
    documents = [tokenize(semantic_text(record)) for record in records]
    count = len(documents)
    frequencies = Counter(word for doc in documents for word in set(doc))
    idf = {
        word: math.log((count + 1) / (frequency + 1)) + 1
        for word, frequency in frequencies.items()
        if frequency >= 2
    }
    vectors = []
    for document in documents:
        tf = Counter(document)
        total = max(len(document), 1)
        vector = {word: (frequency / total) * idf[word] for word, frequency in tf.items() if word in idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        vectors.append({word: value / norm for word, value in vector.items()})
    return vectors, documents


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(word, 0.0) for word, value in left.items())


def semantic_edges(vectors: list[dict[str, float]]) -> list[dict]:
    all_scores: dict[tuple[int, int], float] = {}
    per_node: list[list[tuple[float, int]]] = [[] for _ in vectors]
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            score = cosine(vectors[left], vectors[right])
            if score >= 0.055:
                all_scores[(left, right)] = score
                per_node[left].append((score, right))
                per_node[right].append((score, left))

    selected: set[tuple[int, int]] = set()
    for node, node_scores in enumerate(per_node):
        for _, other in sorted(node_scores, reverse=True)[:7]:
            selected.add((min(node, other), max(node, other)))

    edges = []
    for edge_id, pair in enumerate(sorted(selected)):
        score = round(all_scores[pair], 4)
        edges.append(
            {
                "id": edge_id,
                "from": pair[0],
                "to": pair[1],
                "score": score,
                "width": max(1.0, round(score * 14, 1)),
            }
        )
    return edges


def normalized_average(vectors: list[dict[str, float]], members: list[int]) -> dict[str, float]:
    average: dict[str, float] = defaultdict(float)
    for member in members:
        for word, value in vectors[member].items():
            average[word] += value
    for word in list(average):
        average[word] /= max(len(members), 1)
    norm = math.sqrt(sum(value * value for value in average.values())) or 1.0
    return {word: value / norm for word, value in average.items()}


def semantic_communities(
    records: list[dict],
    vectors: list[dict[str, float]],
    documents: list[list[str]],
) -> tuple[list[int], list[dict]]:
    count = len(records)
    k = min(7, count)
    if not count:
        return [], []

    random.seed(42)
    centers = [vectors[max(range(count), key=lambda index: len(vectors[index]))]]
    while len(centers) < k:
        distances = [
            max(0.0, 1.0 - max(cosine(vector, center) for center in centers))
            for vector in vectors
        ]
        total = sum(distances)
        if total <= 0:
            centers.append(vectors[len(centers) % count])
            continue
        target = random.random() * total
        cumulative = 0.0
        chosen = count - 1
        for index, distance in enumerate(distances):
            cumulative += distance
            if cumulative >= target:
                chosen = index
                break
        centers.append(vectors[chosen])

    assignments = [-1] * count
    for _ in range(50):
        updated = []
        for vector in vectors:
            scores = [cosine(vector, center) for center in centers]
            updated.append(max(range(k), key=lambda index: scores[index]))
        if updated == assignments:
            break
        assignments = updated
        new_centers = []
        for cluster in range(k):
            members = [index for index, value in enumerate(assignments) if value == cluster]
            new_centers.append(
                normalized_average(vectors, members) if members else centers[cluster]
            )
        centers = new_centers

    sizes = Counter(assignments)
    cluster_order = sorted(
        range(k),
        key=lambda cluster: (
            -sizes[cluster],
            cluster,
        ),
    )
    remap = {old: new for new, old in enumerate(cluster_order)}
    assignments = [remap[value] for value in assignments]

    infos = []
    ignored = {
        "physics", "informed", "neural", "network", "networks", "equation",
        "equations", "calculation", "conduction", "ampacity", "soil",
    }
    for cluster in range(k):
        members = [index for index, value in enumerate(assignments) if value == cluster]
        words = Counter(word for index in members for word in documents[index])
        top_words = [
            word
            for word, _ in words.most_common(40)
            if word not in ignored
        ][:3]
        dominant = Counter(
            primary_theme(infer_themes(records[index]), paper_type(records[index]))
            for index in members
        ).most_common(1)[0][0]
        label_tail = ", ".join(top_words) if top_words else THEMES[dominant]["label"]
        infos.append(
            {
                "label": f"G{cluster + 1}: {label_tail}",
                "cat": dominant,
                "count": len(members),
                "color": COMMUNITY_PALETTE[cluster % len(COMMUNITY_PALETTE)],
            }
        )
    return assignments, infos


def author_names(record: dict) -> list[str]:
    authors = [name for kind, name in record["creators"] if kind == "author"]
    if authors:
        return authors
    return [name for _, name in record["creators"]]


def zotero_group_slug(name: str) -> str:
    slug = norm_text(name).replace(" ", "_")
    return slug or "library"


def build_nodes(
    records: list[dict],
    assignments: list[int],
    communities: list[dict],
    metadata: dict,
) -> list[dict]:
    nodes = []
    group_id = metadata.get("group_id")
    group_slug = zotero_group_slug(metadata.get("group_name", ""))
    for index, record in enumerate(records):
        fields = record["fields"]
        title = fields.get("title") or "Sin titulo"
        year = extract_year(fields.get("date", ""))
        authors = author_names(record)
        first_author = authors[0].split()[-1] if authors else "Anonimo"
        themes = infer_themes(record)
        kind = paper_type(record)
        category = primary_theme(themes, kind)
        doi = norm_doi(fields.get("DOI", ""))
        url = fields.get("url", "")
        doi_link = f"https://doi.org/{quote(doi, safe='/():;.-_')}" if doi else url
        attachment = record["attachments"][0] if record["attachments"] else {}
        local_pdf = attachment.get("local_uri", "")

        zotero_cloud = ""
        if group_id:
            base = f"https://www.zotero.org/groups/{group_id}/{group_slug}/items/{record['item_key']}"
            if attachment:
                zotero_cloud = f"{base}/attachment/{attachment['key']}/reader"
            else:
                zotero_cloud = base

        tags = sorted(
            record["tags"],
            key=lambda tag: (
                0 if tag.startswith("tema-") else
                1 if tag.startswith("tipo-") else
                2 if tag.startswith(("metodo-", "propiedad-", "material-", "objeto-", "condicion-")) else
                3,
                norm_text(tag),
            ),
        )
        tag_keys = {norm_text(tag).replace(" ", "-") for tag in tags}
        if "thesis-core" in tag_keys:
            size = 24
        elif kind == "estado_arte":
            size = 17
        elif kind in {"fuente_clasica", "normativa"}:
            size = 15
        elif "tema-pinn" in themes:
            size = 14
        else:
            size = 11

        publication = (
            fields.get("publicationTitle")
            or fields.get("bookTitle")
            or fields.get("publisher")
            or fields.get("university")
            or ""
        )
        community = assignments[index]
        nodes.append(
            {
                "id": index,
                "zotero_item_id": record["item_id"],
                "zotero_key": record["item_key"],
                "label": f"{first_author}\n{year}",
                "title_full": title,
                "year": year,
                "authors": "; ".join(authors[:6]) + (" et al." if len(authors) > 6 else ""),
                "category": category,
                "themes": themes,
                "community": community,
                "comm_label": communities[community]["label"],
                "tags": tags,
                "collections": record["collections"],
                "abstract": (fields.get("abstractNote") or "")[:900].replace("\n", " "),
                "doi_link": doi_link,
                "local_pdf": local_pdf,
                "zotero_cloud": zotero_cloud,
                "pub": publication,
                "color": THEMES[category]["color"],
                "comm_color": communities[community]["color"],
                "size": size,
                "aporte": record["aporte"],
                "paper_type": kind,
                "item_type": record["item_type"],
            }
        )
    return nodes


def js_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def safe_dom_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", value.lower())


def category_legend(nodes: list[dict]) -> str:
    counts = Counter(theme for node in nodes for theme in node["themes"])
    parts = []
    for key, info in sorted(THEMES.items(), key=lambda item: item[1]["order"]):
        if not counts[key]:
            continue
        parts.append(
            f'<div class="li" id="lcat_{safe_dom_id(key)}" onclick="toggleCatF(\'{key}\')">'
            f'<span class="ld" style="background:{info["color"]}"></span>'
            f'<span class="lt">{info["label"]}</span>'
            f'<span class="lc">{counts[key]}</span></div>'
        )
    return "".join(parts)


def community_legend(communities: list[dict]) -> str:
    return "".join(
        f'<div class="li" id="lcomm_{index}" onclick="toggleCommF({index})">'
        f'<span class="ld" style="background:{info["color"]}"></span>'
        f'<span class="lt" style="font-size:.83em">{info["label"]}</span>'
        f'<span class="lc">{info["count"]}</span></div>'
        for index, info in enumerate(communities)
        if info["count"]
    )


def type_legend(nodes: list[dict]) -> str:
    counts = Counter(node["paper_type"] for node in nodes)
    return "".join(
        f'<div class="li" id="ltipo_{key}" onclick="toggleTipoF(\'{key}\')">'
        f'<span class="ld" style="background:{info["color"]}"></span>'
        f'<span class="lt">{info["label"]}</span>'
        f'<span class="lc">{counts[key]}</span></div>'
        for key, info in PAPER_TYPES.items()
        if counts[key]
    )


def replace_constant(source: str, name: str, next_name: str, value: object) -> str:
    pattern = rf"const {name}=.*?;\n(?=const {next_name}=)"
    replacement = f"const {name}={js_json(value)};\n"
    updated, count = re.subn(
        pattern,
        lambda _: replacement,
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f"No se pudo reemplazar const {name}")
    return updated


def replace_sidebar(source: str, element_id: str, next_id: str, content: str) -> str:
    pattern = (
        rf'(<div class="ls" id="{element_id}"[^>]*>).*?'
        rf'(</div>\s*<div class="ls" id="{next_id}")'
    )
    replacement = rf"\1{content}\2"
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"No se pudo actualizar el panel {element_id}")
    return updated


def update_html(
    html_path: Path,
    nodes: list[dict],
    edges: list[dict],
    communities: list[dict],
) -> None:
    source = html_path.read_text(encoding="utf-8")
    source = replace_constant(source, "NODES", "PAPER_TYPES", nodes)
    source = replace_constant(source, "PAPER_TYPES", "EDGES", PAPER_TYPES)
    source = replace_constant(source, "EDGES", "CATS", edges)
    category_data = {
        key: {
            "label": info["label"],
            "color": info["color"],
            "order": info["order"],
        }
        for key, info in THEMES.items()
    }
    source = replace_constant(source, "CATS", "CINFOS", category_data)
    source = replace_constant(source, "CINFOS", "CPAL", communities)

    source = replace_sidebar(source, "scat", "scomm", category_legend(nodes))
    source = replace_sidebar(source, "scomm", "stipo", community_legend(communities))
    type_pattern = (
        r'(<div class="ls" id="stipo"[^>]*>).*?'
        r'(</div>\s*</div>\s*<div id="cw">)'
    )
    source, count = re.subn(
        type_pattern,
        rf"\1{type_legend(nodes)}\2",
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("No se pudo actualizar el panel stipo")

    source = source.replace(
        "const KW_META_PFXS=['coleccion-','tipo-'];",
        "const KW_META_PFXS=['coleccion-','tipo-','tema-','tesis-mia-'];",
        1,
    )
    source = source.replace(
        "onclick=\"swTab('cat')\">Categorias</div>",
        "onclick=\"swTab('cat')\">Temas Zotero</div>",
        1,
    )

    filter_function = """function filterBy(type,val){
  document.getElementById('fb').style.display='block';
  const matches=n=>type==='cat'?(n.themes||[n.category]).includes(val):type==='comm'?n.community===val:n.paper_type===val;
  nodesDS.update(NODES.map(n=>({id:n.id,hidden:!matches(n)})));
  edgesDS.update(EDGES.map(e=>({id:e.id,hidden:!matches(NODES[e.from])||!matches(NODES[e.to])})));
  net.fit({animation:true});updStats();syncKwVis();
}"""
    source, count = re.subn(
        r"function filterBy\(type,val\)\{.*?\n\}",
        filter_function,
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("No se pudo actualizar filterBy")

    source = source.replace(
        "n.tags.some(t=>t.toLowerCase().includes(q))||n.year.includes(q)",
        "n.tags.some(t=>t.toLowerCase().includes(q))||n.collections.some(c=>c.toLowerCase().includes(q))||n.pub.toLowerCase().includes(q)||n.year.includes(q)",
        1,
    )

    show_detail = """function showD(d,nid){
  const themeKeys=d.themes?.length?d.themes:[d.category];
  const themeBadges=themeKeys.map(key=>{const info=CATS[key]||{};return `<span class="bk" style="background:${info.color||'#999'}">${info.label||key}</span>`;}).join('');
  const doi=d.doi_link?`<a href="${d.doi_link}" target="_blank" class="lb lb1">DOI / URL</a>`:'';
  const pdf=d.local_pdf?`<a href="${d.local_pdf}" target="_blank" class="lb lb2">PDF local</a>`:'';
  const cld=d.zotero_cloud?`<a href="${d.zotero_cloud}" target="_blank" class="lb lb3">&#x2601; Zotero</a>`:'';
  const tH=d.tags.filter(Boolean).slice(0,24).map(t=>`<span class="tg">${t}</span>`).join('');
  const apH=d.aporte?`<div class="dap"><div class="dapl">Aporte a la investigación</div>${fixMoji(d.aporte)}</div>`:'';
  const cns=net.getConnectedNodes(nid);
  let nbH='';
  if(cns.length){
    const its=cns.slice(0,14).map(nid2=>{const cn=nodesDS.get(nid2)?._data;if(!cn)return'';const au=cn.authors.split(';')[0].trim().split(' ').pop();return`<li onclick="jumpTo(${nid2})" title="${fixMoji(cn.title_full)}">${au} (${cn.year}) - ${fixMoji(cn.title_full).substring(0,50)}${fixMoji(cn.title_full).length>50?'...':''}</li>`;}).join('');
    nbH=`<details class="nb"><summary>${cns.length} papers conectados semanticamente</summary><ul>${its}</ul></details>`;
  }
  document.getElementById('db').innerHTML=`
    <div class="dt">${fixMoji(d.title_full)}</div>
    <div class="dm"><strong>Autores:</strong> ${fixMoji(d.authors)||'Desconocido'}</div>
    <div class="dm"><strong>Año:</strong> ${d.year}${d.pub?' &middot; <em>'+d.pub+'</em>':''}</div>
    <div style="margin:8px 0">${themeBadges}<span class="bk" style="background:${d.comm_color}">${d.comm_label}</span>${ptBadge(d.paper_type)}</div>
    <div style="margin:6px 0">${doi}${pdf}${cld}</div>
    ${d.abstract?`<div class="da">${fixMoji(d.abstract)}${d.abstract.length>=900?'...':''}</div>`:'<div style="font-size:.74em;color:#aaa;font-style:italic;margin:6px 0">Sin abstract disponible</div>'}
    ${apH}
    <div class="tw">${tH}</div>
    ${nbH}
    <div style="font-size:.68em;color:#bbb;margin-top:10px">Colecciones: ${d.collections.join(', ')||'ninguna'}</div>`;
}"""
    source, count = re.subn(
        r"function showD\(d,nid\)\{.*?\n\}",
        show_detail,
        source,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError("No se pudo actualizar showD")

    html_path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Ruta explicita a zotero.sqlite")
    parser.add_argument("--html", default="zotero_network.html", help="HTML que se actualizara")
    parser.add_argument("--root-collection", default=ROOT_COLLECTION)
    args = parser.parse_args()

    db = locate_zotero_db(args.db)
    html_path = Path(args.html).resolve()
    records, metadata = extract_records(db, args.root_collection)
    records = deduplicate(records)
    vectors, documents = tfidf_vectors(records)
    assignments, communities = semantic_communities(records, vectors, documents)
    nodes = build_nodes(records, assignments, communities, metadata)
    edges = semantic_edges(vectors)
    update_html(html_path, nodes, edges, communities)

    summary = {
        "database": str(db),
        "storage": metadata["storage"],
        "raw_items": metadata["raw_items"],
        "unique_papers": len(nodes),
        "duplicates_merged": metadata["raw_items"] - len(nodes),
        "papers_with_local_pdf": sum(bool(node["local_pdf"]) for node in nodes),
        "papers_with_aporte": sum(bool(node["aporte"]) for node in nodes),
        "semantic_edges": len(edges),
        "theme_memberships": dict(Counter(theme for node in nodes for theme in node["themes"])),
        "paper_types": dict(Counter(node["paper_type"] for node in nodes)),
        "output": str(html_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
