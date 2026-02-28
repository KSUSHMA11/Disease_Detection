/* ============================================================
   AgriVision AI — script.js
   Nature-inspired UI with Falling Leaves + Full Interactivity
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

    /* ── DOM refs ── */
    const imageInput = document.getElementById('imageInput');
    const dropZone = document.getElementById('dropZone');
    const dzIdle = document.getElementById('dzIdle');
    const dzPreview = document.getElementById('dzPreview');
    const imagePreview = document.getElementById('imagePreview');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const plantNameIn = document.getElementById('plantName');
    const predictBtn = document.getElementById('predictBtn');
    const modelTabs = document.querySelectorAll('.mtab');
    const modelInput = document.getElementById('modelType');
    const resetBtn = document.getElementById('resetBtn');
    const retryBtn = document.getElementById('retryBtn');
    const clearHistBtn = document.getElementById('clearHistBtn');
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');
    const navbar = document.getElementById('navbar');

    /* Results states */
    const resIdle = document.getElementById('resIdle');
    const resLoading = document.getElementById('resLoading');
    const resContent = document.getElementById('resContent');
    const resError = document.getElementById('resError');

    /* Result fields */
    const resPlant = document.getElementById('resPlant');
    const resDisease = document.getElementById('resDisease');
    const resConfidence = document.getElementById('resConfidence');
    const confFill = document.getElementById('confFill');
    const resCause = document.getElementById('resCause');
    const resCure = document.getElementById('resCure');
    const resPrevention = document.getElementById('resPrevention');
    const statusBadge = document.getElementById('statusBadge');
    const modelBadge = document.getElementById('modelBadge');

    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');

    let selectedFile = null;

    /* ===========================================================
       🍃 FALLING LEAVES ANIMATION
    =========================================================== */
    const leafPool = ['🍃', '🌿', '🍂', '🍁', '🌱', '☘️', '🌾'];
    const leavesWrap = document.getElementById('leavesContainer');
    const LEAF_COUNT = 18;

    function createLeaf() {
        const el = document.createElement('div');
        el.className = 'leaf';
        el.textContent = leafPool[Math.floor(Math.random() * leafPool.length)];

        const size = 14 + Math.random() * 18; // px font-size
        const leftPct = Math.random() * 100;      // horizontal start %
        const duration = 8 + Math.random() * 14;   // fall speed (s)
        const delay = -Math.random() * 20;       // stagger (negative = start mid-fall)
        const drift = (Math.random() - .5) * 120; // horizontal drift px

        el.style.fontSize = size + 'px';
        el.style.left = leftPct + '%';
        el.style.animationDuration = duration + 's';
        el.style.animationDelay = delay + 's';

        /* Custom fall keyframe with horizontal drift */
        const keyframes = `
      @keyframes leafFall_${el.dataset.id = Math.random().toString(36).slice(2)} {
        0%   { transform: translateY(-60px) translateX(0) rotate(0deg) scale(1); opacity: 0; }
        5%   { opacity: .65; }
        50%  { transform: translateY(50vh) translateX(${drift * .6}px) rotate(180deg) scale(.85); }
        85%  { opacity: .35; }
        100% { transform: translateY(110vh) translateX(${drift}px) rotate(360deg) scale(.5); opacity: 0; }
      }`;
        const style = document.createElement('style');
        style.textContent = keyframes;
        document.head.appendChild(style);
        el.style.animationName = `leafFall_${el.dataset.id}`;

        leavesWrap.appendChild(el);
    }

    for (let i = 0; i < LEAF_COUNT; i++) createLeaf();

    /* ===========================================================
       NAVBAR — scroll shadow + mobile toggle
    =========================================================== */
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 24);

        /* Active nav link highlight */
        const sections = ['hero', 'detect', 'features', 'history'];
        sections.forEach(id => {
            const sec = document.getElementById(id);
            const link = document.querySelector(`.nav-link[href="#${id}"]`);
            if (sec && link) {
                const rect = sec.getBoundingClientRect();
                const active = rect.top <= 100 && rect.bottom > 100;
                link.classList.toggle('active', active);
            }
        });
    }, { passive: true });

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('open');
    });

    /* Smooth scroll for all anchor links */
    document.querySelectorAll('a[href^="#"]').forEach(a => {
        a.addEventListener('click', e => {
            const target = document.querySelector(a.getAttribute('href'));
            if (target) { e.preventDefault(); target.scrollIntoView({ behavior: 'smooth' }); }
            navLinks.classList.remove('open');
        });
    });

    /* ===========================================================
       FILE UPLOAD — click + drag & drop
    =========================================================== */
    dropZone.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', e => {
        const f = e.target.files[0];
        if (f) handleFile(f);
    });

    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-active');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-active'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-active');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) handleFile(f);
    });

    removeImgBtn.addEventListener('click', e => {
        e.stopPropagation();
        clearImage();
    });

    function handleFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = ev => {
            imagePreview.src = ev.target.result;
            dzIdle.style.display = 'none';
            dzPreview.classList.add('shown');
            updateBtn();
            showState('idle');
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        selectedFile = null;
        imageInput.value = '';
        imagePreview.src = '';
        dzPreview.classList.remove('shown');
        dzIdle.style.display = 'flex';
        updateBtn();
    }

    function updateBtn() {
        predictBtn.disabled = !(selectedFile && plantNameIn.value.trim());
    }

    plantNameIn.addEventListener('input', updateBtn);

    /* ===========================================================
       MODEL TABS
    =========================================================== */
    modelTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            modelTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            modelInput.value = tab.dataset.model;
        });
    });

    /* ===========================================================
       PREDICT
    =========================================================== */
    predictBtn.addEventListener('click', runPrediction);

    async function runPrediction() {
        if (!selectedFile || !plantNameIn.value.trim()) return;

        showState('loading');
        predictBtn.disabled = true;

        await animateSteps();

        const fd = new FormData();
        fd.append('file', selectedFile);
        const model = modelInput.value;
        const url = `http://localhost:8000/predict?model_type=${model}`;

        try {
            const res = await fetch(url, { method: 'POST', body: fd });
            if (!res.ok) throw new Error('Non-200');
            const data = await res.json();

            if (data.error) throw new Error(data.error);

            populateResults(data, model);
            showState('content');
            saveHistory(data, model);
            renderHistory();
        } catch (err) {
            console.error(err);
            showState('error');
        } finally {
            predictBtn.disabled = false;
        }
    }

    /* ===========================================================
       LOADING STEPS
    =========================================================== */
    function animateSteps() {
        return new Promise(resolve => {
            const steps = [step1, step2, step3];
            steps.forEach(s => s.classList.remove('active'));
            let i = 0;
            const iv = setInterval(() => {
                if (i > 0) steps[i - 1].classList.remove('active');
                if (i < steps.length) { steps[i].classList.add('active'); i++; }
                else { clearInterval(iv); resolve(); }
            }, 650);
        });
    }

    /* ===========================================================
       POPULATE RESULTS
    =========================================================== */
    function populateResults(data, model) {
        resPlant.textContent = data.plant_name || plantNameIn.value.trim() || 'Unknown Plant';
        resDisease.textContent = data.disease || 'Unknown Disease';
        resCause.textContent = data.cause || 'No data available.';
        resCure.textContent = data.cure || 'No data available.';
        resPrevention.textContent = data.prevention || 'No data available.';
        modelBadge.textContent = model === 'vit' ? 'ViT' : 'Swin v2';

        const conf = parseFloat(data.confidence) || 0;
        resConfidence.textContent = conf + '%';
        setTimeout(() => { confFill.style.width = conf + '%'; }, 80);

        const healthy = (data.disease || '').toLowerCase().includes('healthy');
        statusBadge.textContent = healthy ? '✓ Healthy' : '⚠ Diseased';
        statusBadge.className = 'rbadge status-badge' + (healthy ? ' healthy-badge' : '');
    }

    /* ===========================================================
       RESULT STATE MACHINE
    =========================================================== */
    function showState(state) {
        resIdle.style.display = state === 'idle' ? 'flex' : 'none';
        resLoading.style.display = state === 'loading' ? 'flex' : 'none';
        resContent.style.display = state === 'content' ? 'flex' : 'none';
        resError.style.display = state === 'error' ? 'flex' : 'none';
    }

    /* ===========================================================
       RESET / RETRY
    =========================================================== */
    resetBtn.addEventListener('click', () => {
        clearImage();
        confFill.style.width = '0%';
        showState('idle');
    });

    retryBtn.addEventListener('click', () => {
        if (selectedFile) runPrediction();
        else showState('idle');
    });

    /* ===========================================================
       HISTORY
    =========================================================== */
    function saveHistory(data, model) {
        let h = getHistory();
        h.unshift({
            plant: data.plant_name || plantNameIn.value.trim() || 'Unknown',
            disease: data.disease || 'Unknown',
            confidence: data.confidence || '0',
            model,
            date: new Date().toLocaleString()
        });
        if (h.length > 10) h = h.slice(0, 10);
        localStorage.setItem('agrivision_hist', JSON.stringify(h));
    }

    function getHistory() {
        try { return JSON.parse(localStorage.getItem('agrivision_hist') || '[]'); } catch { return []; }
    }

    function renderHistory() {
        const list = document.getElementById('histList');
        const empty = document.getElementById('histEmpty');
        const h = getHistory();

        list.querySelectorAll('.hist-entry').forEach(el => el.remove());

        if (!h.length) { empty.style.display = 'flex'; return; }
        empty.style.display = 'none';

        h.forEach(e => {
            const healthy = (e.disease || '').toLowerCase().includes('healthy');
            const div = document.createElement('div');
            div.className = 'hist-entry';
            div.innerHTML = `
        <span class="hist-emoji">${healthy ? '🌿' : '🔴'}</span>
        <div class="hist-info">
          <p class="hist-plant">${esc(e.plant)}</p>
          <p class="hist-disease">${esc(e.disease)}</p>
        </div>
        <div class="hist-meta">
          <span class="hist-conf">${esc(String(e.confidence))}%</span>
          <span class="hist-date">${esc(e.date)}</span>
        </div>`;
            list.appendChild(div);
        });
    }

    clearHistBtn.addEventListener('click', () => {
        localStorage.removeItem('agrivision_hist');
        renderHistory();
    });

    renderHistory();

    /* ===========================================================
       INTERSECTION OBSERVER — fade-in sections
    =========================================================== */
    const io = new IntersectionObserver(entries => {
        entries.forEach(en => {
            if (en.isIntersecting) {
                en.target.style.opacity = '1';
                en.target.style.transform = 'translateY(0)';
                io.unobserve(en.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.detect-card, .feat-card, .hist-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity .6s ease, transform .6s ease';
        io.observe(el);
    });

    /* ===========================================================
       UTILITY
    =========================================================== */
    function esc(str) {
        return String(str)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

});
