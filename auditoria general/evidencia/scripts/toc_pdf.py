import fitz, sys
d = fitz.open(sys.argv[1])
print('paginas', len(d))
maxlev = int(sys.argv[2]) if len(sys.argv) > 2 else 3
for lev, title, page in d.get_toc():
    if lev <= maxlev:
        print('  ' * (lev - 1) + title[:95], '|', page)
