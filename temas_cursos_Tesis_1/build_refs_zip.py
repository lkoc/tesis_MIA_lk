"""
Busca en Zotero los PDFs de las referencias citadas en la tesis
y genera un ZIP en la carpeta temas_cursoi_Tesis_1.
"""
import sqlite3, shutil, os, zipfile

# --- Configuracion ---
ZOTERO_DB   = r"C:\Users\QU1267\Zotero\zotero.sqlite"
ZOTERO_STOR = r"C:\Users\QU1267\Zotero\storage"
TEMP_DB     = r"C:\Users\QU1267\AppData\Local\Temp\zotero_copy.sqlite"
OUT_ZIP     = r"c:\usr\ths_mia_fiis\tesis_MIA_lk\temas_cursoi_Tesis_1\refs_tesis.zip"

# Tuplas: (apellido_o_vacio, anio, fragmento_titulo)
# Si apellido esta vacio se omite el chequeo de autor
TARGETS = [
    ("Neher",       "1957", "temperature rise",       "Neher_1957"),
    ("",            "2023", "iec 60287",               "IEC_2023"),       # IEC 2023 -> title only
    ("Anders",      "2005", "unfavorable",             "Anders_2005"),
    ("Aras",        "2005", "ampacity",                "Aras_2005"),
    ("Kim",         "2025", "bedding material",        "Kim_2025"),
    ("Khumalo",     "2025", "drying out",              "Khumalo_2025"),
    ("Enescu",      "2021", "dynamic thermal rating",  "Enescu_2021"),
    ("Fariz",       "2026", "systematic mapping",      "Fariz_2026"),
    ("Raissi",      "2019", "physics-informed neural", "Raissi_2019"),
    ("Lawal",       "2022", "evolution and beyond",    "Lawal_2022"),     # fragmento corregido
    ("Incropera",   "2011", "heat and mass transfer",  "Incropera_2011"),
    ("Oclon",       "2015", "cable bedding",           "Oclon_2015"),
    ("Al-Dulaimi",  "2024", "FEM",                     "Al-Dulaimi_2024"),
    ("Atoccsa",     "2024", "ampacity",                "Atoccsa_2024"),
    ("Mobius",      "2025", "seasonal",                "Mobius_2025"),
    ("Billah",      "2023", "inverse heat transfer",   "Billah_2023"),
    ("CIGRE",       "2022", "rating examples",         "CIGRE_2022"),
    ("",            "2002", "iec 60853",               "IEC_2002"),       # IEC 2002 -> title only
    ("Kolawole",    "2024", "thermal conductivity",    "Kolawole_2024"),
    ("CIGRE",       "2025", "finite element",          "CIGRE_2025"),
    ("Pan",         "2025", "temperature field",       "Pan_2025"),
]

shutil.copy2(ZOTERO_DB, TEMP_DB)
con = sqlite3.connect(TEMP_DB)
cur = con.cursor()

cur.execute("""
SELECT i.itemID, i.key,
  (SELECT c2.lastName FROM creators c2
   JOIN itemCreators ic2 ON ic2.creatorID=c2.creatorID
   WHERE ic2.itemID=i.itemID ORDER BY ic2.orderIndex LIMIT 1) as first_author,
  MAX(CASE WHEN f.fieldName='date'  THEN dv.value END) as date_val,
  MAX(CASE WHEN f.fieldName='title' THEN dv.value END) as title
FROM items i
LEFT JOIN itemData d ON d.itemID=i.itemID
LEFT JOIN itemDataValues dv ON dv.valueID=d.valueID
LEFT JOIN fields f ON f.fieldID=d.fieldID
WHERE i.itemTypeID NOT IN (1,3,28)
GROUP BY i.itemID
""")
items = cur.fetchall()

def norm(s):
    if s is None: return ""
    s = s.lower()
    for a, b in [("\u00e4","a"), ("\u00f6","o"), ("\u00fc","u"),
                 ("\u0142","l"), ("\u0144","n"), ("\u0119","e"),
                 ("\u00f3","o"), ("\u00e1","a"), ("\u00e9","e"),
                 ("\u00ed","i"), ("\u00fa","u"), ("-","")]:
        s = s.replace(a, b)
    return s

def match_item(item, tgt_author, tgt_year, tgt_frag, label):
    _,_,author,date_val,title = item
    year_str = str(date_val or "")[:4]
    if tgt_author and norm(tgt_author) not in norm(author or ""):
        return False
    if tgt_year and tgt_year != year_str:
        return False
    if tgt_frag and norm(tgt_frag) not in norm(title or ""):
        return False
    return True

found_items = {}
for tgt in TARGETS:
    tgt_author, tgt_year, tgt_frag, label = tgt
    for item in items:
        if match_item(item, tgt_author, tgt_year, tgt_frag, label):
            found_items[label] = (item[0], item[1], item[4])
            break

print(f"\n--- Items encontrados ({len(found_items)}/{len(TARGETS)}) ---")
for lbl,(iid,key,ttl) in found_items.items():
    print(f"  {lbl:25s} itemID={iid}  {str(ttl)[:55]}")

missing = [t for t in TARGETS if t[3] not in found_items]
if missing:
    print(f"\n--- NO encontrados ({len(missing)}) ---")
    for t in missing: print(f"  {t[3]} (autor='{t[0]}' year={t[1]} frag='{t[2]}')")

cur.execute("""
SELECT ia.parentItemID, i.key, ia.path, ia.contentType
FROM itemAttachments ia JOIN items i ON i.itemID=ia.itemID
WHERE ia.contentType='application/pdf'
""")
attachments = cur.fetchall()
con.close()

att_map = {}
for (pid,akey,apath,ctype) in attachments:
    att_map.setdefault(pid,[]).append((akey,apath))

print("\n--- Buscando PDFs en disco ---")
pdf_files=[]
for lbl,(iid,ikey,ttl) in found_items.items():
    found_pdf=None
    for (akey,apath) in att_map.get(iid,[]):
        if apath and apath.startswith("storage:"):
            fname=apath[len("storage:"):]
            fpath=os.path.join(ZOTERO_STOR,akey,fname)
        else:
            fpath=None
            folder=os.path.join(ZOTERO_STOR,akey)
            if os.path.isdir(folder):
                for fn in os.listdir(folder):
                    if fn.lower().endswith(".pdf"):
                        fpath=os.path.join(folder,fn); break
        if fpath and os.path.isfile(fpath):
            found_pdf=fpath; break
    if not found_pdf:
        folder=os.path.join(ZOTERO_STOR,ikey)
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower().endswith(".pdf"):
                    found_pdf=os.path.join(folder,fn); break
    if found_pdf:
        print(f"  OK  {lbl:25s} {os.path.basename(found_pdf)}")
        pdf_files.append((lbl,found_pdf))
    else:
        print(f"  --  {lbl:25s} sin PDF en disco")

print(f"\n--- Generando ZIP: {OUT_ZIP} ---")
with zipfile.ZipFile(OUT_ZIP,'w',zipfile.ZIP_DEFLATED) as zf:
    for (lbl,fpath) in pdf_files:
        arcname=f"{lbl}__{os.path.basename(fpath)}"
        zf.write(fpath,arcname)
        print(f"  + {arcname}")
print(f"\nZIP generado con {len(pdf_files)} PDFs.")
