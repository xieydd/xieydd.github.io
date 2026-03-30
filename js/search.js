(function () {
  var fuse = null;
  var searchData = null;
  var modal = document.getElementById('search-modal');
  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  if (!modal || !input || !results) return;

  var lang = document.documentElement.lang || 'en';
  var isZh = lang.startsWith('zh');

  function open() {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
    setTimeout(function () { input.focus(); }, 50);
  }

  function close() {
    modal.classList.remove('active');
    document.body.style.overflow = '';
    input.value = '';
    results.innerHTML = '<div class="search-hint">' + (isZh ? '输入关键词开始搜索' : 'Type to start searching') + '</div>';
  }

  function loadFuse(callback) {
    if (fuse) return callback();
    var base = document.querySelector('base');
    var root = base ? base.getAttribute('href') : '/';
    var jsonUrl = root + 'index.json';

    fetch(jsonUrl)
      .then(function (res) { return res.json(); })
      .then(function (data) {
        searchData = data;
        fuse = new Fuse(data, {
          keys: [
            { name: 'title', weight: 0.4 },
            { name: 'tags', weight: 0.3 },
            { name: 'categories', weight: 0.2 },
            { name: 'contents', weight: 0.1 }
          ],
          threshold: 0.35,
          ignoreLocation: true,
          useExtendedSearch: false,
          minMatchCharLength: 2
        });
        callback();
      })
      .catch(function (err) {
        console.error('Failed to load search index:', err);
        results.innerHTML = '<div class="search-hint">' + (isZh ? '搜索索引加载失败' : 'Failed to load search index') + '</div>';
      });
  }

  function render(items) {
    if (items.length === 0) {
      results.innerHTML = '<div class="search-hint">' + (isZh ? '没有找到相关文章' : 'No results found') + '</div>';
      return;
    }

    var html = '<ul class="search-results-list">';
    items.forEach(function (item) {
      var title = item.item.title;
      var permalink = item.item.permalink;
      var date = item.item.date;
      var tags = (item.item.tags || []).slice(0, 3);

      html += '<li class="search-result-item">';
      html += '<a href="' + permalink + '" class="search-result-link">';
      html += '<span class="search-result-title">' + title + '</span>';
      html += '<span class="search-result-meta">';
      html += '<time>' + date + '</time>';
      if (tags.length > 0) {
        html += ' · ' + tags.map(function (t) { return '#' + t; }).join(' ');
      }
      html += '</span>';
      html += '</a>';
      html += '</li>';
    });
    html += '</ul>';
    results.innerHTML = html;
  }

  // Debounce
  var debounceTimer = null;
  input.addEventListener('input', function () {
    clearTimeout(debounceTimer);
    var query = input.value.trim();
    if (query.length < 2) {
      results.innerHTML = '<div class="search-hint">' + (isZh ? '输入关键词开始搜索' : 'Type to start searching') + '</div>';
      return;
    }
    debounceTimer = setTimeout(function () {
      loadFuse(function () {
        render(fuse.search(query));
      });
    }, 150);
  });

  // Click on result link should close modal (for swup compat)
  results.addEventListener('click', function (e) {
    var link = e.target.closest('a');
    if (link) {
      close();
    }
  });

  // Close on backdrop click
  modal.querySelector('.search-modal-backdrop').addEventListener('click', close);

  // Close on ESC
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      close();
    }
    // Ctrl/Cmd + K to open
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      if (modal.classList.contains('active')) {
        close();
      } else {
        open();
      }
    }
  });

  // Expose open for button click
  window.openSearch = open;
  window.closeSearch = close;
})();
