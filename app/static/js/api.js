/**
 * api.js — CineMatch shared JS utilities
 * Loaded on every page. Provides:
 *   - apiFetch(url, method, body)  → fetch wrapper with auth redirect
 *   - showToast(msg, type)         → toast notification
 *   - showSpinner() / hideSpinner()
 */

// ──────────────────────────────────────
// Core fetch wrapper
// ──────────────────────────────────────
async function apiFetch(url, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    credentials: 'same-origin',
  };
  if (body && method !== 'GET') {
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (res.status === 401) {
    window.location.href = '/';
    return res;
  }
  return res;
}

/**
 * Parse JSON error body from a failed response (4xx/5xx).
 * Use after checking !res.ok.
 */
async function parseApiError(res, fallback = 'Request failed') {
  try {
    const j = await res.json();
    if (j && (j.error || j.message)) return j.error || j.message;
  } catch (e) { /* not JSON */ }
  if (res.status === 503) {
    return 'Service temporarily unavailable. If recommendations fail, run: python recommender.py --build';
  }
  if (res.status === 404) return 'Not found';
  if (res.status === 400) return 'Invalid request';
  return fallback;
}

// ──────────────────────────────────────
// Toast notifications
// ──────────────────────────────────────
function showToast(message, type = 'success') {
  // Remove existing toasts
  document.querySelectorAll('.cm-toast').forEach(t => t.remove());

  const colors = {
    success: 'bg-surface-container-highest border-primary-container/50 text-on-surface',
    error:   'bg-error-container/30 border-error/50 text-on-error-container',
    info:    'bg-tertiary-container/30 border-tertiary/50 text-on-tertiary-container',
  };

  const icons = { success: 'check_circle', error: 'error', info: 'info' };

  const toast = document.createElement('div');
  toast.className = `cm-toast fixed bottom-24 md:bottom-lg right-lg z-[90] flex items-center gap-sm px-lg py-md rounded-xl border shadow-lg backdrop-blur-md transition-all duration-300 translate-y-4 opacity-0 ${colors[type] || colors.success}`;
  toast.innerHTML = `
    <span class="material-symbols-outlined text-[20px]">${icons[type] || 'check_circle'}</span>
    <span class="font-label-bold text-label-bold">${message}</span>
  `;
  document.body.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-4', 'opacity-0');
  });

  // Auto-remove after 3s
  setTimeout(() => {
    toast.classList.add('translate-y-4', 'opacity-0');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ──────────────────────────────────────
// Loading spinner
// ──────────────────────────────────────
function showSpinner(containerId = null) {
  const spinnerHtml = `
    <div id="cm-spinner" class="flex flex-col items-center justify-center gap-lg py-xxl">
      <div class="w-12 h-12 rounded-full border-4 border-surface-container-high border-t-primary-container animate-spin"></div>
      <p class="text-on-surface-variant font-body-md text-body-md">Loading…</p>
    </div>`;
  if (containerId) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = spinnerHtml;
  } else {
    document.body.insertAdjacentHTML('beforeend',
      `<div id="cm-spinner-overlay" class="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm flex items-center justify-center">${spinnerHtml}</div>`
    );
  }
}

function hideSpinner(containerId = null) {
  const id = containerId ? 'cm-spinner' : 'cm-spinner-overlay';
  document.getElementById(id)?.remove();
}

// ──────────────────────────────────────
// Movie card builder (reused by discover + movie_detail)
// ──────────────────────────────────────
function buildMovieCard(movie) {
  const matchPct  = Math.round((movie.hybrid_score || 0) * 100);
  let genreList = movie.genres || [];
  if (typeof genreList === 'string') {
    genreList = genreList.split('|').map(s => s.trim()).filter(Boolean);
  }
  if (!Array.isArray(genreList)) genreList = [];
  const genres    = genreList.slice(0, 2).join(' · ');
  const year      = movie.year || '';
  const poster    = movie.tmdb_poster_url || '/static/img/no_poster.png';
  const isLatest   = movie.rec_source === 'latest';
  const isRegional = movie.rec_source === 'regional_in';
  const isPopular  = movie.rec_source === 'tmdb_popular';

  return `
    <article
      class="group relative bg-surface-container-low border border-surface-variant/40 rounded-xl overflow-hidden
             cursor-pointer transition-all duration-300 hover:scale-[1.04] hover:shadow-[0_0_24px_rgba(229,9,20,0.25)]
             hover:border-primary-container/50"
      onclick="window.location.href='/movie/${movie.movie_id}'"
    >
      <!-- Poster -->
      <div class="aspect-[2/3] bg-surface-container-highest overflow-hidden relative">
        <img
          src="${poster}"
          alt="${movie.title}"
          class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          onerror="this.src='/static/img/no_poster.png'"
        />
        ${isLatest ? `
        <div class="absolute top-sm left-sm px-sm py-xs rounded border border-tertiary/50 bg-tertiary/15 backdrop-blur-sm
                    text-tertiary font-label-bold text-label-sm">Newer in catalog</div>` : ''}
        ${isRegional ? `
        <div class="absolute top-sm left-sm px-sm py-xs rounded border border-primary-container/40 bg-primary-container/10 backdrop-blur-sm
                    text-primary-container font-label-bold text-label-sm">Regional pick</div>` : ''}
        ${isPopular ? `
        <div class="absolute top-sm left-sm px-sm py-xs rounded border border-secondary/40 bg-secondary/10 backdrop-blur-sm
                    text-secondary font-label-bold text-label-sm">Popular pick</div>` : ''}
        <!-- Match badge (visible on hover) -->
        ${matchPct > 0 ? `
        <div class="absolute top-sm right-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300
                    bg-background/80 backdrop-blur-sm rounded-full px-sm py-xs border border-primary-container/50">
          <span class="text-primary-container font-label-bold text-label-sm">${matchPct}% match</span>
        </div>` : ''}
      </div>
      <!-- Info -->
      <div class="p-md">
        <h3 class="font-label-bold text-label-bold text-on-surface truncate">${movie.title}</h3>
        <p class="font-label-sm text-label-sm text-on-surface-variant mt-xs">${year} ${genres ? '· ' + genres : ''}</p>
      </div>
    </article>`;
}

/**
 * TMDb similar/recommendations row: optional in-app id; else link to TMDb.
 * Title/poster are escaped for safe innerHTML.
 */
function escapeHtmlAttr(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function buildTmdbRelatedCard(x) {
  const title = x.title || 'Untitled';
  const year = x.year != null && x.year !== '' ? x.year : '';
  const poster = (x.poster_url && x.poster_url.trim()) ? x.poster_url : '/static/img/no_poster.png';
  const inApp = x.in_app_movie_id != null && x.in_app_movie_id !== undefined;
  const tmdbId = x.tmdb_id;
  const badge = inApp
    ? 'In CineMatch'
    : 'TMDb';
  const badgeClass = inApp
    ? 'bg-primary-container/20 text-primary-container border-primary-container/40'
    : 'bg-surface-container-highest/80 text-on-surface-variant border-surface-variant/60';
  const onClick = inApp
    ? `window.location.href='/movie/${x.in_app_movie_id}'`
    : `window.open('https://www.themoviedb.org/movie/${tmdbId}','_blank','noopener')`;

  return `
    <article
      class="group relative bg-surface-container-low border border-surface-variant/40 rounded-xl overflow-hidden
             cursor-pointer transition-all duration-300 hover:scale-[1.04] hover:shadow-[0_0_24px_rgba(229,9,20,0.2)]
             hover:border-tertiary/40"
      onclick="${onClick}"
    >
      <div class="aspect-[2/3] bg-surface-container-highest overflow-hidden relative">
        <img
          src="${escapeHtmlAttr(poster)}"
          alt="${escapeHtmlAttr(title)}"
          class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          onerror="this.src='/static/img/no_poster.png'"
        />
        <div class="absolute top-sm left-sm px-sm py-xs rounded border text-label-sm font-label-bold ${badgeClass} backdrop-blur-sm">
          ${escapeHtmlAttr(badge)}
        </div>
      </div>
      <div class="p-md">
        <h3 class="font-label-bold text-label-bold text-on-surface truncate">${escapeHtmlAttr(title)}</h3>
        <p class="font-label-sm text-label-sm text-on-surface-variant mt-xs">${year ? escapeHtmlAttr(year) : '—'}</p>
      </div>
    </article>`;
}

// ──────────────────────────────────────
// Autocomplete for movie search inputs
// options: { useSuggest: true } = local + TMDb (Filters page)
// ──────────────────────────────────────
function initMovieAutocomplete(inputEl, onSelect, options) {
  const useSuggest = options && options.useSuggest === true;
  let debounceTimer;
  let dropdown;

  inputEl.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const q = inputEl.value.trim();
    if (q.length < 2) { dropdown?.remove(); return; }

    debounceTimer = setTimeout(async () => {
      const url = useSuggest
        ? `/api/movies/suggest?q=${encodeURIComponent(q)}`
        : `/api/movies/search?q=${encodeURIComponent(q)}`;
      let res, payload;
      try {
        res = await apiFetch(url);
        if (!res.ok) return;
        payload = await res.json();
      } catch (_) { return; }
      const data = useSuggest ? (payload.results || []) : payload;

      dropdown?.remove();
      if (!data.length) {
        dropdown = document.createElement('ul');
        dropdown.className =
          'absolute z-30 w-full bg-surface-container-high border border-surface-variant rounded-xl ' +
          'shadow-xl mt-xs overflow-hidden p-md text-on-surface-variant font-body-sm text-sm';
        dropdown.textContent = useSuggest
          ? 'No results in library or TMDb. Try another spelling.'
          : 'No movies match that search';
        const wrapper = inputEl.closest('.relative') || inputEl.parentElement;
        wrapper.style.position = 'relative';
        wrapper.appendChild(dropdown);
        return;
      }

      dropdown = document.createElement('ul');
      dropdown.className =
        'absolute z-30 w-full bg-surface-container-high border border-surface-variant rounded-xl ' +
        'shadow-xl mt-xs overflow-hidden max-h-[300px] overflow-y-auto';

      data.forEach(m => {
        const li = document.createElement('li');
        li.className =
          'flex items-center gap-md px-md py-sm hover:bg-surface-container-highest cursor-pointer ' +
          'transition-colors duration-150';
        const badge = m.source === 'tmdb'
          ? (m.in_catalog
              ? '<span class="text-[10px] uppercase text-primary-container font-bold shrink-0">TMDb · in app</span>'
              : '<span class="text-[10px] uppercase text-on-surface-variant font-bold shrink-0">TMDb · not in app</span>')
          : '<span class="text-[10px] uppercase text-on-surface-variant font-bold shrink-0">Library</span>';
        const y = m.year != null && m.year !== undefined ? m.year : '';
        li.innerHTML = `
          <span class="material-symbols-outlined text-on-surface-variant text-[18px] shrink-0">movie</span>
          <div class="min-w-0 flex-1">
            <div class="font-body-md text-body-md text-on-surface truncate">${m.title}</div>
            <div class="flex items-center justify-between gap-sm mt-0.5">${badge}</div>
          </div>
          <span class="font-label-sm text-label-sm text-on-surface-variant shrink-0">${y}</span>`;
        li.addEventListener('click', () => {
          inputEl.value = m.title;
          dropdown.remove();
          if (!onSelect) return;
          const ret = onSelect(m);
          if (ret && typeof ret.then === 'function') {
            ret.catch((err) => {
              if (window.showToast) {
                const msg = (err && err.message) || 'Something went wrong';
                showToast(String(msg), 'error');
              }
            });
          }
        });
        dropdown.appendChild(li);
      });

      const wrapper = inputEl.closest('.relative') || inputEl.parentElement;
      wrapper.style.position = 'relative';
      wrapper.appendChild(dropdown);
    }, 300);
  });

  // Close dropdown on outside click
  document.addEventListener('click', (e) => {
    if (!inputEl.contains(e.target)) dropdown?.remove();
  });
}

// ──────────────────────────────────────
// Discover — chip filters (client-side on loaded rows)
// ──────────────────────────────────────
function discoverMovieMatchesFilter(m, key) {
  const raw = m.genres;
  const gs = (Array.isArray(raw)
    ? raw.join('|')
    : typeof raw === 'string' ? raw : '').toLowerCase();
  const year = m.year;
  const ar = m.avg_rating;
  const hs = m.hybrid_score;
  switch (key) {
    case 'all':
      return true;
    case 'action':
      return gs.includes('action') || gs.includes('adventure');
    case 'scifi':
      return gs.includes('sci-fi') || gs.includes('thriller');
    case 'acclaimed':
      if (ar != null) return ar >= 3.7;
      if (hs != null) return hs >= 0.35;
      return true;
    case 'new':
      if (year == null) return false;
      if (year >= 2000) return true;
      return year >= 1995;
    default:
      return true;
  }
}

function filterDiscoverMovies(movies, key) {
  return (movies || []).filter((m) => discoverMovieMatchesFilter(m, key));
}

// ──────────────────────────────────────
// Global header: search modal + notification stub
// ──────────────────────────────────────
function initHeaderChrome() {
  const modal = document.getElementById('cm-search-modal');
  const input = document.getElementById('cm-global-movie-search');
  if (modal && input) {
    const close = () => {
      modal.classList.add('hidden');
      input.value = '';
    };
    const open = () => {
      modal.classList.remove('hidden');
      setTimeout(() => input.focus(), 50);
    };
    const backdrop = modal.querySelector('[data-cm-search-backdrop]');
    backdrop?.addEventListener('click', close);
    modal.querySelector('[data-cm-search-close]')?.addEventListener('click', close);
    document.getElementById('cm-header-search-desk')?.addEventListener('click', open);
    document.getElementById('cm-header-search-mobile')?.addEventListener('click', open);
    initMovieAutocomplete(input, (m) => {
      close();
      if (m && m.movie_id != null) {
        window.location.href = '/movie/' + m.movie_id;
      } else if (m && m.in_catalog === false) {
        showToast('That title is not in the CineMatch library. Pick a Library match or use TMDb \u00b7 in app.', 'info');
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !modal.classList.contains('hidden')) close();
    });
  }
  const notif = document.getElementById('cm-header-notifications');
  notif?.addEventListener('click', () => {
    showToast('No new notifications', 'info');
  });
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHeaderChrome);
  } else {
    initHeaderChrome();
  }
}
