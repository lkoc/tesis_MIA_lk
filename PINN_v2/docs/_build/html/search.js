
function plain(s) { return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase(); }
var box = document.getElementById('q'), hits = document.getElementById('hits');
box.addEventListener('input', function () {
  var words = plain(box.value).split(/\s+/).filter(Boolean);
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
