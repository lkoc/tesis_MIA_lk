import fitz

doc = fitz.open("Enescu_2020_thermal_assessment.pdf")

keywords = [
    'is defined', 'defined as', 'refers to', 'denotes',
    'ampacity', 'current rating', 'thermal resistance',
    'thermal capacit', 'heat flux', 'temperature rise',
    'conductor temp', 'IEC 60287', 'iec 60853',
    'electrothermal', 'analogy', 'heat transfer', 'fourier',
    'dynamic thermal', 'backfill', 'joule', 'dielectric loss',
    'skin effect', 'proximity effect'
]

print("=" * 70)
for pg in range(min(15, len(doc))):
    text = doc[pg].get_text()
    lines = text.split('\n')
    for i, line in enumerate(lines):
        ll = line.lower()
        if any(kw in ll for kw in keywords) and len(line.strip()) > 50:
            context = ' '.join(lines[max(0, i):i+4]).replace('\n', ' ').strip()
            if len(context) > 60:
                print(f"p{pg+1}: {context[:280]}")
                print()
