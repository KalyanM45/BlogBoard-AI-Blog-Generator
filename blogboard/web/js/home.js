/**
 * home.js — Home page logic
 * Uses async blogs-data.js to load articles dynamically.
 */
document.addEventListener('DOMContentLoaded', async () => {
    initNav();
    initParticles();
    await Promise.all([loadRecentPosts(), loadStats()]);
});

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

/* ── Stats Counter ── */
async function loadStats() {
    const total = await getTotalCount();
    animateCounter('totalBlogs', total);
    animateCounter('totalCategories', 6);
}

function animateCounter(id, target) {
    const container = document.getElementById(id);
    if (!container) return;
    const el = container.querySelector('.stat-num') || container;
    let current = 0;
    const step = Math.ceil(target / 40);
    const timer = setInterval(() => {
        current += step;
        if (current >= target) { current = target; clearInterval(timer); }
        el.textContent = current;
    }, 40);
}

/* ── Recent Posts ── */
async function loadRecentPosts() {
    const container = document.getElementById('recentPosts');
    if (!container) return;

    container.innerHTML = '<p style="color:var(--text-muted);padding:20px">Loading articles...</p>';

    const recents = await getRecentBlogs(6);

    if (recents.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted);padding:20px">No articles published yet. Check back soon!</p>';
        return;
    }

    container.innerHTML = recents.map(blog => {
        const meta = CATEGORY_META[blog.category];
        return `
    <a href="post.html#id=${encodeURIComponent(blog.id)}" class="recent-card">
      <div class="recent-card-meta">
        <span class="recent-cat-badge" style="background:${meta.bgColor};color:${meta.color}">
          ${meta.shortLabel}
        </span>
        <span class="recent-date">${formatDate(blog.date)}</span>
      </div>
      <h3 class="recent-title">${escapeHtml(blog.title)}</h3>
      <p class="recent-desc">${escapeHtml(blog.description)}</p>
      <span class="recent-readtime">📖 ${blog.readTime} read</span>
    </a>`;
    }).join('');
}

function escapeHtml(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* ── Particle Background ─────────────────────────────── */
function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H, particles = [];

    function resize() {
        W = canvas.width = canvas.offsetWidth;
        H = canvas.height = canvas.offsetHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const N = 60;
    for (let i = 0; i < N; i++) {
        particles.push({
            x: Math.random() * W, y: Math.random() * H,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            a: Math.random() * 0.5 + 0.1,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        particles.forEach(p => {
            p.x = (p.x + p.vx + W) % W;
            p.y = (p.y + p.vy + H) % H;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(124, 106, 247, ${p.a})`;
            ctx.fill();
        });

        // Draw connecting lines between nearby particles
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(124, 106, 247, ${0.12 * (1 - dist / 100)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
}
