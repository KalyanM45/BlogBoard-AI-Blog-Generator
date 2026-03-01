/**
 * category.js — Category page logic
 * Reads category from hash (#cat=ml) and loads articles dynamically from articles.json.
 */
let currentSort = 'newest';
let searchTerm = '';
let catKey = '';
let cachedBlogs = [];

document.addEventListener('DOMContentLoaded', async () => {
    catKey = getHashParam('cat') || 'ml';
    initNav();
    applyCategoryTheme(catKey);
    await renderBlogList();

    // Search
    document.getElementById('searchInput')?.addEventListener('input', async e => {
        searchTerm = e.target.value.toLowerCase().trim();
        await renderBlogList();
    });

    // Hash change (nav link clicks between categories)
    window.addEventListener('hashchange', async () => {
        const newCat = getHashParam('cat') || 'ml';
        if (newCat !== catKey) {
            catKey = newCat;
            updateNavHighlight(catKey);
            applyCategoryTheme(catKey);
            cachedBlogs = [];
            searchTerm = '';
            const si = document.getElementById('searchInput');
            if (si) si.value = '';
            await renderBlogList();
        }
    });
});

/* ── Parse hash params ── */
function getHashParam(key) {
    const hash = window.location.hash.replace(/^#/, '');
    return new URLSearchParams(hash).get(key);
}

/* ── Nav ── */
function initNav() {
    const navbar = document.getElementById('navbar');
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');

    updateNavHighlight(catKey);

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 30);
    });

    hamburger?.addEventListener('click', () => {
        hamburger.classList.toggle('open');
        navLinks.classList.toggle('open');
    });
}

function updateNavHighlight(cat) {
    document.querySelectorAll('.nav-link[data-cat]').forEach(link => {
        link.classList.toggle('active', link.getAttribute('data-cat') === cat);
    });
}

/* ── Domain schedule info ── */
const DOMAIN_SCHEDULE = {
    ml: { day: 'Monday', time: '8 AM IST' },
    dl: { day: 'Tuesday', time: '8 AM IST' },
    statistics: { day: 'Wednesday', time: '8 AM IST' },
    nlp: { day: 'Thursday', time: '8 AM IST' },
    cv: { day: 'Friday', time: '8 AM IST' },
    genai: { day: 'Saturday', time: '8 AM IST' },
    ainews: { day: 'Sunday', time: '8 AM IST' },
};

/* ── Sort ── */
function setSortOrder(order) {
    currentSort = order;
    document.getElementById('sortNewest')?.classList.toggle('active', order === 'newest');
    document.getElementById('sortOldest')?.classList.toggle('active', order === 'oldest');
    renderBlogList();
}

/* ── Apply Category Theme ── */
function applyCategoryTheme(cat) {
    const meta = CATEGORY_META[cat];
    if (!meta) return;

    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    document.body.classList.add(`theme-${cat}`);
    document.title = `${meta.label} — BlogBoard`;

    const icon = document.getElementById('catHeroIcon');
    const badge = document.getElementById('catBadge');
    const title = document.getElementById('catHeroTitle');
    const desc = document.getElementById('catHeroDesc');
    const scheduleBadge = document.getElementById('catScheduleBadge');

    if (icon) icon.textContent = meta.icon;
    if (badge) { badge.textContent = meta.shortLabel; badge.style.background = meta.bgColor; badge.style.color = meta.color; }
    if (title) title.textContent = meta.label;
    if (desc) desc.textContent = meta.description;

    if (scheduleBadge) {
        const sched = DOMAIN_SCHEDULE[cat];
        if (sched && sched.day) {
            scheduleBadge.innerHTML = `
                <span class="sched-icon">🗓️</span>
                <span>Fresh articles drop every <strong>${sched.day}</strong> — live by <strong>${sched.time}</strong></span>
            `;
            scheduleBadge.style.display = 'flex';
        } else {
            scheduleBadge.innerHTML = `<span class="sched-icon">📡</span><span>Published as breaking news arrives</span>`;
            scheduleBadge.style.display = 'flex';
        }
    }
}

/* ── Render Blog List ── */
async function renderBlogList() {
    const listEl = document.getElementById('blogList');
    const emptyEl = document.getElementById('emptyState');
    const countEl = document.getElementById('catPostCount');
    if (!listEl) return;

    listEl.innerHTML = '<p style="color:var(--text-muted);padding:20px">Loading articles...</p>';

    let blogs = await getBlogsByCategory(catKey, currentSort);
    if (countEl) countEl.textContent = blogs.length;

    // Search filter
    if (searchTerm) {
        blogs = blogs.filter(b =>
            b.title.toLowerCase().includes(searchTerm) ||
            b.description.toLowerCase().includes(searchTerm) ||
            (b.tags || []).some(t => t.toLowerCase().includes(searchTerm))
        );
    }

    if (blogs.length === 0) {
        listEl.innerHTML = '';
        emptyEl?.classList.remove('hidden');
        return;
    }

    emptyEl?.classList.add('hidden');
    const meta = CATEGORY_META[catKey];

    listEl.innerHTML = blogs.map((blog, idx) => `
    <a href="post.html#id=${encodeURIComponent(blog.id)}" class="blog-item">
      <span class="blog-item-num">${String(idx + 1).padStart(2, '0')}</span>
      <div class="blog-item-body">
        <div class="blog-item-meta">
          <span class="blog-item-date">${formatDate(blog.date)}</span>
          ${(blog.tags || []).slice(0, 2).map(tag =>
        `<span class="blog-item-tag" style="background:${meta.bgColor};color:${meta.color}">#${tag}</span>`
    ).join('')}
        </div>
        <h2 class="blog-item-title">${escapeHtml(blog.title)}</h2>
        <p class="blog-item-desc">${escapeHtml(blog.description)}</p>
        <div class="blog-item-footer">
          <span class="read-time">📖 ${blog.readTime} read</span>
        </div>
      </div>
      <span class="blog-item-arrow">→</span>
    </a>
  `).join('');
}

function escapeHtml(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
