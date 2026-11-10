"""Builds the documentation as static HTML, with no tools beyond pygments (already installed).

Run from anywhere:  python docs/build_docs.py
Then open docs/_build/html/index.html in a browser; nothing needs a server or the internet.
"""
import html
import json
import re
import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

HERE = Path(__file__).resolve().parent
OUT = HERE.joinpath('_build', 'html')
PROJECT = 'PINN_v2'

# Reading order of the pages; the sidebar and the previous/next links follow it.
PAGES = ['index', 'quickstart', 'architecture', 'cases', 'models', 'training', 'search', 'fem',
         'evaluation', 'reproducibility', 'cli', 'frontend', 'results', 'limitations']

# Just enough TeX for the formulas used in these pages.
TEX = {r'\nabla': '\u2207', r'\cdot': '\u00b7', r'\partial': '\u2202', r'\infty': '\u221e', r'\Delta': '\u0394',
       r'\lambda': '\u03bb', r'\,': '\u2009', r'\le': '\u2264', r'\ge': '\u2265', r'\approx': '\u2248'}

CSS = """
body { font: 16px Georgia, 'Times New Roman', serif; color: #3e4349; margin: 0; background: #fff; }
a { color: #004b6b; text-decoration: none; border-bottom: 1px dotted #004b6b; }
a:hover { color: #6d4100; border-bottom: 1px solid #6d4100; }
.layout { display: flex; max-width: 1100px; margin: 0 auto; }
.sidebar { width: 250px; flex: none; padding: 30px 20px 30px 10px; font-size: 14px; line-height: 1.4; }
.sidebar h1 { font-size: 26px; margin: 0 0 10px 0; font-weight: normal; }
.sidebar h3 { font-size: 17px; font-weight: normal; margin: 22px 0 6px 0; color: #444; }
.sidebar ul { list-style: none; margin: 0; padding: 0; }
.sidebar li { margin: 3px 0; }
.sidebar li.current > a { font-weight: bold; border-bottom: none; }
.sidebar ul ul { margin-left: 14px; font-size: 13px; }
.sidebar input { width: 90%; padding: 4px; font: inherit; border: 1px solid #aaa; }
#hits li { margin: 6px 0; }
#hits small { color: #777; display: block; }
.content { flex: 1; min-width: 0; padding: 20px 30px 40px 30px; line-height: 1.5; }
.content h1 { font-size: 36px; font-weight: normal; margin: 10px 0 20px 0; }
.content h2 { font-size: 27px; font-weight: normal; margin: 34px 0 12px 0; }
.content h3 { font-size: 21px; font-weight: normal; margin: 26px 0 10px 0; }
.content p, .content li { text-align: left; }
code { font: 14px Consolas, 'Courier New', monospace; background: #ecebe7; padding: 1px 4px; }
.highlight { background: #eee; border: 1px solid #ddd; margin: 14px 0; padding: 4px 12px; overflow-x: auto; }
.highlight pre { margin: 6px 0; font: 14px Consolas, 'Courier New', monospace; line-height: 1.4; }
table { border-collapse: collapse; margin: 14px 0; font-size: 15px; }
th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: left; vertical-align: top; }
th { background: #f3f3f0; }
blockquote { margin: 14px 0; padding: 4px 16px; border-left: 4px solid #ccc; background: #f8f8f6; }
.math { font-style: italic; }
.pager { margin-top: 40px; padding-top: 10px; border-top: 1px solid #ddd; display: flex; justify-content: space-between; font-size: 15px; }
.footer { color: #888; font-size: 13px; margin-top: 30px; }
@media (max-width: 800px) { .layout { display: block; } .sidebar { width: auto; } }
"""

SCRIPT = """
function plain(s) { return s.normalize('NFD').replace(/[\\u0300-\\u036f]/g, '').toLowerCase(); }
var box = document.getElementById('q'), hits = document.getElementById('hits');
box.addEventListener('input', function () {
  var words = plain(box.value).split(/\\s+/).filter(Boolean);
  hits.innerHTML = '';
  if (!words.length) { return; }
  var found = [];
  INDEX.forEach(function (page) {
    var text = plain(page.text), score = 0;
    words.forEach(function (w) { var at = text.indexOf(w); if (at < 0) { score = -1e9; } else { score += 1 + (plain(page.title).indexOf(w) >= 0 ? 5 : 0); } });
    if (score > 0) { found.push([score, page]); }
  });
  found.sort(function (a, b) { return b[0] - a[0]; });
  if (!found.length) { hits.innerHTML = '<li>Sin resultados</li>'; }
  found.forEach(function (f) {
    var at = plain(f[1].text).indexOf(words[0]);
    var snippet = f[1].text.substring(Math.max(0, at - 40), at + 70);
    var li = document.createElement('li');
    li.innerHTML = '<a href="' + f[1].name + '.html">' + f[1].title + '</a><small></small>';
    li.querySelector('small').textContent = '...' + snippet + '...';
    hits.appendChild(li);
  });
});
"""


def tex(text):
    for command, symbol in TEX.items():
        text = text.replace(command, symbol)
    text = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', text)
    return re.sub(r'_(\w|\u221e)', r'<sub>\1</sub>', text)


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', re.sub(r'<[^>]+>', '', text).lower()).strip('-')


def inline(text, links):
    """Markdown inline syntax to HTML. Code and math are set aside first so their characters stay literal."""
    stash = []

    def keep(markup):
        stash.append(markup)
        return '\x00%d\x00' % (len(stash) - 1)

    text = re.sub(r'`([^`]+)`', lambda m: keep('<code>' + html.escape(m.group(1)) + '</code>'), text)
    text = re.sub(r'\$([^$\n]+)\$', lambda m: keep('<span class="math">' + tex(html.escape(m.group(1))) + '</span>'), text)
    text = html.escape(text, quote=False)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?![\w*])', r'<em>\1</em>', text)

    def link(m):
        label, target = m.group(1), m.group(2)
        if target.endswith('.md') or '.md#' in target:
            links.append(target.split('#')[0][:-3])
            target = target.replace('.md', '.html')
        return '<a href="%s">%s</a>' % (target, label)

    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)
    return re.sub('\x00(\\d+)\x00', lambda m: stash[int(m.group(1))], text)


def code_block(body, lang):
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except Exception:
        lexer = TextLexer()
    marked = highlight(body, lexer, HtmlFormatter(nowrap=True))
    return '<div class="highlight"><pre>%s</pre></div>' % marked


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def convert(source):
    """Returns (html, title, h2 list, broken-link targets) for one page."""
    lines, out, toc, links = source.split('\n'), [], [], []
    title, i = '', 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('```'):
            lang, body = line[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                body.append(lines[i])
                i += 1
            i += 1
            out.append(code_block('\n'.join(body), lang))
            continue
        m = re.match(r'(#{1,4}) (.*)', line)
        if m:
            level, text = len(m.group(1)), inline(m.group(2), links)
            if level == 1 and not title:
                title = re.sub(r'<[^>]+>', '', text)
            if level == 2:
                toc.append((slug(text), re.sub(r'<[^>]+>', '', text)))
            out.append('<h%d id="%s">%s</h%d>' % (level, slug(text), text, level))
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append(split_row(lines[i]))
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            table = '<table><thead><tr>%s</tr></thead><tbody>' % ''.join('<th>%s</th>' % inline(c, links) for c in head)
            for row in body:
                table += '<tr>%s</tr>' % ''.join('<td>%s</td>' % inline(c, links) for c in row)
            out.append(table + '</tbody></table>')
            continue
        if re.match(r'\s*([-*]|\d+\.) ', line):
            ordered = re.match(r'\s*\d+\. ', line) is not None
            items = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(('```', '|', '#')):
                item = re.match(r'\s*(?:[-*]|\d+\.) (.*)', lines[i])
                if item:
                    items.append(item.group(1))
                else:
                    items[-1] += ' ' + lines[i].strip()
                i += 1
            tag = 'ol' if ordered else 'ul'
            out.append('<%s>%s</%s>' % (tag, ''.join('<li>%s</li>' % inline(t, links) for t in items), tag))
            continue
        if line.startswith('>'):
            quote = []
            while i < len(lines) and lines[i].startswith('>'):
                quote.append(lines[i][1:].strip())
                i += 1
            out.append('<blockquote><p>%s</p></blockquote>' % inline(' '.join(quote), links))
            continue
        if not line.strip():
            i += 1
            continue
        paragraph = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('```', '|', '>', '#')) and not re.match(r'\s*([-*]|\d+\.) ', lines[i]):
            paragraph.append(lines[i].strip())
            i += 1
        out.append('<p>%s</p>' % inline(' '.join(paragraph), links))
    return '\n'.join(out), title, toc, links


def plain_text(source):
    text = re.sub(r'```.*?```', ' ', source, flags=re.S)
    text = re.sub(r'[`#|>*$]', ' ', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return re.sub(r'\s+', ' ', text).strip()


def page_html(name, body, title, toc, titles, order):
    nav = []
    for other in order:
        if other == name:
            sub = ''.join('<li><a href="#%s">%s</a></li>' % (anchor, html.escape(label)) for anchor, label in toc)
            nav.append('<li class="current"><a href="%s.html">%s</a>%s</li>' % (other, html.escape(titles[other]), '<ul>%s</ul>' % sub if sub else ''))
        else:
            nav.append('<li><a href="%s.html">%s</a></li>' % (other, html.escape(titles[other])))
    at = order.index(name)
    pager = '<div class="pager"><span>%s</span><span>%s</span></div>' % (
        '&laquo; <a href="%s.html">%s</a>' % (order[at - 1], html.escape(titles[order[at - 1]])) if at > 0 else '',
        '<a href="%s.html">%s</a> &raquo;' % (order[at + 1], html.escape(titles[order[at + 1]])) if at + 1 < len(order) else '')
    return """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>%s - %s</title>
<link rel="stylesheet" href="style.css"></head><body>
<div class="layout">
<div class="sidebar"><h1><a href="index.html" style="border:none">%s</a></h1>
<h3>Navegaci\u00f3n</h3><ul>%s</ul>
<h3>B\u00fasqueda r\u00e1pida</h3><input id="q" type="text" placeholder="buscar..." autocomplete="off"><ul id="hits"></ul>
</div>
<div class="content">%s%s<div class="footer">Documentaci\u00f3n generada con docs/build_docs.py; funciona sin conexi\u00f3n.</div></div>
</div>
<script src="search_index.js"></script><script src="search.js"></script>
</body></html>""" % (html.escape(title), PROJECT, PROJECT, ''.join(nav), body, pager)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages, index = {}, []
    for name in PAGES:
        source = HERE.joinpath(name + '.md').read_text(encoding='utf-8')
        pages[name] = convert(source)
        index.append(dict(name=name, title=pages[name][1], text=plain_text(source)))
    titles = {name: pages[name][1] for name in PAGES}
    for name, (body, title, toc, links) in pages.items():
        for target in links:
            if target not in PAGES:
                print('warning: %s links to missing page %s' % (name, target))
        OUT.joinpath(name + '.html').write_text(page_html(name, body, title, toc, titles, PAGES), encoding='utf-8')
    css = CSS + HtmlFormatter(style='default').get_style_defs('.highlight')
    OUT.joinpath('style.css').write_text(css, encoding='utf-8')
    OUT.joinpath('search.js').write_text(SCRIPT, encoding='utf-8')
    OUT.joinpath('search_index.js').write_text('var INDEX = %s;' % json.dumps(index, ensure_ascii=False), encoding='utf-8')
    print('built %d pages in %s' % (len(PAGES), OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())