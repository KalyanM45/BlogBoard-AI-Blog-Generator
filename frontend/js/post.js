/**
 * post.js — Blog post viewer with markdown rendering
 * Uses window.location.hash (#id=...) to avoid server stripping query params
 */
document.addEventListener('DOMContentLoaded', () => {
    initNav();
    loadPost();
    initReadingProgress();
});

/* ── Parse hash params ── */
function getHashParam(key) {
    // Supports formats: #id=foo  OR  #cat=ml&id=foo
    const hash = window.location.hash.replace(/^#/, '');
    const params = new URLSearchParams(hash);
    return params.get(key);
}

/* ── Nav ── */
function initNav() {
    const navbar = document.getElementById('navbar');
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');

    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 30);
    });

    hamburger?.addEventListener('click', () => {
        hamburger.classList.toggle('open');
        navLinks.classList.toggle('open');
    });
}

/* ── Reading Progress ── */
function initReadingProgress() {
    const bar = document.getElementById('readingProgress');
    if (!bar) return;
    window.addEventListener('scroll', () => {
        const doc = document.documentElement;
        const scrollTop = doc.scrollTop || document.body.scrollTop;
        const scrollHeight = doc.scrollHeight - doc.clientHeight;
        bar.style.width = scrollHeight > 0 ? `${(scrollTop / scrollHeight) * 100}%` : '0%';
    });
}

/* ── Load Post ── */
async function loadPost() {
    // Read id from hash: post.html#id=blogs/ml/intro-to-ml.md
    const rawId = getHashParam('id');
    const id = rawId ? decodeURIComponent(rawId) : null;
    const contentEl = document.getElementById('postContent');

    if (!id) {
        // No id in hash — redirect to home
        window.location.replace('index.html');
        return;
    }

    const blog = await getBlogById(id);
    if (!blog) {
        showError('Post not found. It may have been removed or the link is incorrect.', contentEl);
        return;
    }
    const meta = CATEGORY_META[blog.category];

    // Set page title
    document.title = `${blog.title} — BlogBoard`;

    // Breadcrumb
    const catLink = document.getElementById('catLink');
    if (catLink) {
        catLink.textContent = meta.label;
        catLink.href = `category.html#cat=${blog.category}`;
    }
    const postTitleSpan = document.getElementById('postTitle');
    if (postTitleSpan) postTitleSpan.textContent = blog.title;

    // Header
    document.getElementById('postTitleH1').textContent = blog.title;
    document.getElementById('postDate').textContent = formatDate(blog.date);

    const catBadge = document.getElementById('postCatBadge');
    if (catBadge) {
        catBadge.textContent = meta.shortLabel;
        catBadge.className = `post-cat-badge badge-${blog.category}`;
    }

    const readTimeEl = document.getElementById('postReadTime');
    if (readTimeEl) readTimeEl.textContent = `📖 ${blog.readTime} read`;

    // Active nav link
    document.querySelectorAll('.nav-link[href]').forEach(link => {
        if (link.href.includes(`cat=${blog.category}`)) link.classList.add('active');
    });

    // Back link
    const backBtn = document.getElementById('backToCat');
    if (backBtn) {
        backBtn.href = `category.html#cat=${blog.category}`;
        backBtn.textContent = `← Back to ${meta.shortLabel}`;
    }

    // Tags
    const tagsEl = document.getElementById('postTags');
    if (tagsEl && blog.tags?.length) {
        tagsEl.innerHTML = blog.tags.map(t =>
            `<span class="post-tag">#${t}</span>`
        ).join('');
    }

    // Fetch and render markdown from R2
    try {
        const response = await fetch(`${R2_PUBLIC_URL}/${blog.file}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const mdText = await response.text();
        renderMarkdown(mdText, contentEl);
        buildTOC();
    } catch (err) {
        showError(
            `Could not load the article file.<br>
       <small>Expected URL: <code>${R2_PUBLIC_URL}/${blog.file}</code></small><br>
       <small>Ensure your R2 bucket is public and the URL in blogs-data.js is correct.</small>`,
            contentEl
        );
        console.error('Failed to load blog post:', err);
    }
}

/* ── Render Markdown ── */
function renderMarkdown(mdText, container) {
    marked.setOptions({
        gfm: true,
        breaks: true,
    });

    // Add IDs to headings for TOC
    const renderer = new marked.Renderer();
    renderer.heading = (text, level) => {
        const escapedText = (typeof text === 'object' ? text.text : text)
            .toLowerCase().replace(/[^\w]+/g, '-');
        const rawText = typeof text === 'object' ? text.text : text;
        return `<h${level} id="${escapedText}">${rawText}</h${level}>`;
    };

    container.innerHTML = marked.parse(mdText, { renderer });

    // Syntax highlight all code blocks
    if (window.hljs) {
        container.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    }

    // Make code blocks copy-able
    container.querySelectorAll('pre').forEach(pre => {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.textContent = 'Copy';
        btn.style.cssText = `
      position:absolute; top:10px; right:12px;
      background:rgba(124,106,247,0.15); color:#a89cf7;
      border:1px solid rgba(124,106,247,0.25); border-radius:6px;
      padding:3px 10px; font-size:0.75rem; cursor:pointer;
      font-family:var(--font-sans); transition:all 0.15s;
    `;
        btn.addEventListener('click', () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code?.textContent || '').then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 2000);
            });
        });
        pre.style.position = 'relative';
        pre.appendChild(btn);
    });
}

/* ── Build Table of Contents ── */
function buildTOC() {
    const content = document.getElementById('postContent');
    const tocNav = document.getElementById('tocNav');
    if (!content || !tocNav) return;

    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) return;

    tocNav.innerHTML = '';

    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                tocNav.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
                const activeLink = tocNav.querySelector(`[data-target="${entry.target.id}"]`);
                activeLink?.classList.add('active');
            }
        });
    }, { rootMargin: '-20% 0px -70% 0px' });

    headings.forEach(h => {
        const level = h.tagName.toLowerCase();
        const link = document.createElement('a');
        link.href = `#${h.id}`;
        link.setAttribute('data-target', h.id);
        link.textContent = h.textContent;
        link.className = `toc-link level-${level}`;
        link.addEventListener('click', e => {
            e.preventDefault();
            document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
        tocNav.appendChild(link);
        observer.observe(h);
    });
}

/* ── Error State ── */
function showError(message, container) {
    if (!container) return;
    container.innerHTML = `
    <div style="padding:40px;text-align:center;color:var(--text-muted)">
      <div style="font-size:2.5rem;margin-bottom:16px">⚠️</div>
      <h3 style="color:var(--text-secondary);margin-bottom:12px">Unable to load article</h3>
      <p style="line-height:1.7">${message}</p>
    </div>`;
}
