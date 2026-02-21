/* stv-nav.js — Shared menu button + side menu for all pages */
(function() {
  'use strict';

  var nav = document.querySelector('.stv-nav');
  if (!nav) return;

  var theme = nav.getAttribute('data-stv-theme') || 'light';
  var page = nav.getAttribute('data-stv-page') || 'gallery';

  // ── Menu button ──
  var btn = document.createElement('button');
  btn.className = 'stv-menu-btn';
  if (theme === 'dark') btn.classList.add('stv-menu-btn--dark');
  else if (theme === 'floating') btn.classList.add('stv-menu-btn--floating');
  btn.setAttribute('aria-label', 'Menu');
  btn.innerHTML = '<span class="bar-h"></span><span class="bar-v"></span>';
  document.body.appendChild(btn);

  // ── Side menu ──
  var pages = [
    { label: 'Gallery', href: '/', id: 'gallery' },
    { label: 'Feed', href: '/feed.html', id: 'feed' },
    { label: 'Taste Map', href: '/taste.html', id: 'taste' },
    { label: 'Analytics', href: '/analytics.html', id: 'analytics' },
    { label: 'Data', href: '/data.html', id: 'data' },
    { label: 'About', href: '/about.html', id: 'about' }
  ];

  var linksHtml = '';
  for (var i = 0; i < pages.length; i++) {
    var p = pages[i];
    var cls = 'stv-side-menu-link';
    if (p.id === page) cls += ' current';
    linksHtml += '<a class="' + cls + '" href="' + p.href + '">' + p.label + '</a>';
  }

  var menu = document.createElement('div');
  menu.className = 'stv-side-menu';
  menu.innerHTML =
    '<div class="stv-side-menu-backdrop"></div>' +
    '<div class="stv-side-menu-panel">' +
      linksHtml +
      '<div class="stv-side-menu-footer">' +
        '<a href="#">Contact</a>' +
        '<a href="#">Terms</a>' +
        '<a href="#">Privacy</a>' +
      '</div>' +
    '</div>';
  document.body.appendChild(menu);

  // ── State ──
  function isOpen() { return menu.classList.contains('open'); }

  function open() {
    menu.classList.add('open');
    btn.classList.add('open');
  }

  function close() {
    menu.classList.remove('open');
    btn.classList.remove('open');
  }

  function toggle() {
    if (isOpen()) close(); else open();
  }

  // ── Events ──
  btn.addEventListener('click', toggle);
  menu.querySelector('.stv-side-menu-backdrop').addEventListener('click', close);

  // Current-page link: close menu instead of navigating
  var links = menu.querySelectorAll('.stv-side-menu-link');
  for (var j = 0; j < links.length; j++) {
    (function(link) {
      link.addEventListener('click', function(e) {
        if (link.classList.contains('current')) {
          e.preventDefault();
          close();
        }
      });
    })(links[j]);
  }

  // Escape key — capture phase + stopImmediatePropagation
  // so page-level Escape handlers only fire when menu is closed
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && isOpen()) {
      close();
      e.stopImmediatePropagation();
    }
  }, true);

  // ── Expose globals ──
  window.stvMenuBtn = btn;
  window.stvToggleMenu = toggle;
  window.stvIsMenuOpen = isOpen;
})();
