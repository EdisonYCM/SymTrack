window.HELP_IMPROVE_VIDEOJS = false;

// Escape: qualitative compare → benchmark figure → teaser lightbox
document.addEventListener('keydown', function(event) {
    if (event.key !== 'Escape') return;
    const qualLb = document.getElementById('qualCompareLightbox');
    if (qualLb && qualLb.classList.contains('is-open')) {
        event.preventDefault();
        closeQualCompareLightbox();
        return;
    }
    const benchLb = document.getElementById('benchLightbox');
    if (benchLb && benchLb.classList.contains('is-open')) {
        event.preventDefault();
        closeBenchLightbox();
        return;
    }
    const teaserLb = document.getElementById('teaserLightbox');
    const teaserTrigger = document.getElementById('teaserLightboxTrigger');
    if (teaserLb && teaserLb.classList.contains('is-open')) {
        event.preventDefault();
        teaserLb.classList.remove('is-open');
        document.body.classList.remove('teaser-lightbox-open');
        if (teaserTrigger) teaserTrigger.setAttribute('aria-expanded', 'false');
        teaserLb.addEventListener('transitionend', function onTeaserLbEnd(e) {
            if (e.target !== teaserLb || e.propertyName !== 'opacity') return;
            teaserLb.setAttribute('hidden', '');
            teaserLb.removeEventListener('transitionend', onTeaserLbEnd);
            if (teaserTrigger) teaserTrigger.focus();
        });
    }
});

/** All paired filenames under static/images/quan_exam/input and output (T2I first) */
var QUAL_EXAM_FILES = [
    't2i_1.png',
    't2i_2.png',
    't2i_3.png',
    'bbox_1.png',
    'bbox_2.png',
    'doodle_1.png',
    'doodle_2.png',
    'force_1.png',
    'force_2.png',
    'force_3.png',
    'tii_1.png',
    'tii_2.png',
    'tii_3.png',
    'tii_4.png',
    'tii_5.png',
    'traj_1.png',
    'traj_2.jpg',
    'vm_1.png',
    'vm_2.png',
    'vm_3.png'
];

var QUAL_EXAM_PREFIX_LABELS = {
    t2i: 'Text2image',
    bbox: 'Textbbox edit',
    doodle: 'Doodles edit',
    force: 'Force understand',
    tii: 'Text in image edit',
    traj: 'Trajectory understand',
    vm: 'Visual marker edit'
};

function qualExamLabel(filename) {
    var stem = filename.replace(/\.(png|jpe?g)$/i, '');
    var m = stem.match(/^([a-z0-9]+)_(\d+)$/i);
    if (!m) return stem.replace(/_/g, ' ');
    var prefix = m[1].toLowerCase();
    var num = m[2];
    var title = QUAL_EXAM_PREFIX_LABELS[prefix];
    if (!title) return stem.replace(/_/g, ' ');
    return title + ' ' + num;
}

function initQualResultsGrid() {
    var grid = document.getElementById('qualResultsGrid');
    if (!grid) return;
    var base = 'static/images/quan_exam';
    QUAL_EXAM_FILES.forEach(function (file) {
        var label = qualExamLabel(file);
        var inUrl = base + '/input/' + file;
        var outUrl = base + '/output/' + file;
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'qual-result-card';
        btn.setAttribute('data-qual-in', inUrl);
        btn.setAttribute('data-qual-out', outUrl);
        btn.setAttribute('data-qual-label', label);
        btn.setAttribute('aria-label', 'Open compare view for ' + label);
        btn.innerHTML =
            '<span class="qual-result-card-inner">' +
            '<span class="qual-result-preview">' +
            '<span class="qual-result-pair" aria-hidden="true">' +
            '<span class="qual-result-cell"><img src="' +
            inUrl +
            '" alt="" loading="lazy" decoding="async"></span>' +
            '<span class="qual-result-cell"><img src="' +
            outUrl +
            '" alt="" loading="lazy" decoding="async"></span>' +
            '</span>' +
            '</span>' +
            '<span class="qual-result-meta">' +
            '<span class="qual-result-name">' +
            label +
            '</span>' +
            '</span>' +
            '</span>';
        grid.appendChild(btn);
    });
}

function closeQualCompareLightbox() {
    var lightbox = document.getElementById('qualCompareLightbox');
    if (!lightbox || !lightbox.classList.contains('is-open')) return;
    var lastFocus = lightbox._qualReturnFocus;
    lightbox.classList.remove('is-open');
    document.body.classList.remove('qual-compare-open');
    lightbox.addEventListener('transitionend', function onEnd(e) {
        if (e.target !== lightbox || e.propertyName !== 'opacity') return;
        lightbox.setAttribute('hidden', '');
        lightbox.removeEventListener('transitionend', onEnd);
        if (lastFocus && typeof lastFocus.focus === 'function') lastFocus.focus();
        lightbox._qualReturnFocus = null;
    });
}

function initQualCompareLightbox() {
    var lightbox = document.getElementById('qualCompareLightbox');
    var grid = document.getElementById('qualResultsGrid');
    if (!lightbox || !grid) return;

    var closers = lightbox.querySelectorAll('[data-qual-compare-close]');
    var titleEl = document.getElementById('qualCompareTitle');
    var viewport = document.getElementById('qualCompareViewport');
    var clip = document.getElementById('qualCompareClip');
    var handle = document.getElementById('qualCompareHandle');
    var imgIn = document.getElementById('qualCompareInputImg');
    var imgOut = document.getElementById('qualCompareOutputImg');
    if (!viewport || !clip || !imgIn || !imgOut) return;

    function setSplitPct(pct) {
        pct = Math.max(0, Math.min(100, pct));
        clip.style.width = pct + '%';
        handle.style.left = pct + '%';
        handle.setAttribute('aria-valuenow', String(Math.round(pct)));
    }

    function openFromButton(btn) {
        var inSrc = btn.getAttribute('data-qual-in');
        var outSrc = btn.getAttribute('data-qual-out');
        var label = btn.getAttribute('data-qual-label') || '';
        if (!inSrc || !outSrc) return;
        lightbox._qualReturnFocus = btn;
        imgIn.src = inSrc;
        imgOut.src = outSrc;
        imgIn.alt = 'Input — ' + label;
        imgOut.alt = 'Output — ' + label;
        if (titleEl) titleEl.textContent = label;
        setSplitPct(50);
        lightbox.removeAttribute('hidden');
        document.body.classList.add('qual-compare-open');
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                lightbox.classList.add('is-open');
            });
        });
        window.setTimeout(function () {
            if (handle) handle.focus({ preventScroll: true });
        }, 200);
    }

    grid.addEventListener('click', function (e) {
        var btn = e.target.closest('.qual-result-card');
        if (!btn || !grid.contains(btn)) return;
        openFromButton(btn);
    });

    closers.forEach(function (el) {
        el.addEventListener('click', function (e) {
            e.preventDefault();
            closeQualCompareLightbox();
        });
    });

    var dragging = false;

    function pctFromClientX(clientX) {
        var rect = viewport.getBoundingClientRect();
        if (rect.width <= 0) return 50;
        return ((clientX - rect.left) / rect.width) * 100;
    }

    function onPointerMove(e) {
        if (!dragging) return;
        setSplitPct(pctFromClientX(e.clientX));
    }

    function onPointerUp(e) {
        if (!dragging) return;
        dragging = false;
        if (viewport.releasePointerCapture && e.pointerId != null) {
            try {
                viewport.releasePointerCapture(e.pointerId);
            } catch (err) { /* ignore */ }
        }
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
        window.removeEventListener('pointercancel', onPointerUp);
    }

    viewport.addEventListener('pointerdown', function (e) {
        if (!lightbox.classList.contains('is-open')) return;
        if (e.button !== undefined && e.button !== 0) return;
        dragging = true;
        setSplitPct(pctFromClientX(e.clientX));
        if (viewport.setPointerCapture) viewport.setPointerCapture(e.pointerId);
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
        window.addEventListener('pointercancel', onPointerUp);
    });

    if (handle) {
        handle.addEventListener('keydown', function (e) {
            if (!lightbox.classList.contains('is-open')) return;
            var step = e.shiftKey ? 10 : 4;
            var cur = parseFloat(clip.style.width) || 50;
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                setSplitPct(cur - step);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                setSplitPct(cur + step);
            }
        });
    }
}

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');
    
    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            button.classList.add('copied');
            copyText.textContent = 'Cop';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Scroll to top functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Show/hide scroll to top button
window.addEventListener('scroll', function() {
    const scrollButton = document.querySelector('.scroll-to-top');
    if (window.pageYOffset > 300) {
        scrollButton.classList.add('visible');
    } else {
        scrollButton.classList.remove('visible');
    }
});

function closeBenchLightbox() {
    var lb = document.getElementById('benchLightbox');
    if (!lb || !lb.classList.contains('is-open')) return;
    var ret = lb._benchReturnFocus;
    lb.classList.remove('is-open');
    document.body.classList.remove('bench-lightbox-open');
    lb.addEventListener('transitionend', function onEnd(e) {
        if (e.target !== lb || e.propertyName !== 'opacity') return;
        lb.setAttribute('hidden', '');
        lb.removeEventListener('transitionend', onEnd);
        var imgEl = document.getElementById('benchLightboxImg');
        if (imgEl) {
            imgEl.removeAttribute('src');
            imgEl.alt = '';
        }
        if (ret && typeof ret.focus === 'function') ret.focus();
        lb._benchReturnFocus = null;
    });
}

/** VP-Bench table: column-wise best is bolded at render time */
var VP_BENCH = {
    cols: ['C2I', 'T2I', 'TIE', 'FU', 'TBE', 'TU', 'VME', 'DE', 'Total'],
    evaluators: [
        {
            id: 'gpt',
            label: 'GPT-5.2',
            caption: 'Evaluator: GPT-5.2',
            rows: [
                {
                    name: 'Nano Banana',
                    values: [0.68, 0.959, 0.152, 0.127, 0.023, 0.04, 0.136, 0.302, 0.302],
                    ours: false
                },
                {
                    name: 'Omnigen2',
                    values: [0.11, 0.02, 0, 0, 0, 0, 0, 0.023, 0.019],
                    ours: false
                },
                {
                    name: 'Kontext',
                    values: [0.09, 0.02, 0.028, 0.02, 0, 0.08, 0.003, 0.093, 0.042],
                    ours: false
                },
                {
                    name: 'Qwen-IE-2509',
                    values: [0.24, 0.12, 0.08, 0.02, 0.022, 0.06, 0.02, 0.047, 0.076],
                    ours: false
                },
                {
                    name: 'FlowInOne (Ours)',
                    values: [0.85, 0.8, 0.079, 0.5, 0.116, 0.24, 0.083, 0.465, 0.392],
                    ours: true
                }
            ]
        },
        {
            id: 'gemini',
            label: 'Gemini 3',
            caption: 'Evaluator: Gemini 3',
            rows: [
                {
                    name: 'Nano Banana',
                    values: [0.65, 0.98, 0.423, 0.52, 0.614, 0.02, 0.548, 0.721, 0.56],
                    ours: false
                },
                {
                    name: 'Omnigen2',
                    values: [0.02, 0.02, 0.017, 0, 0, 0, 0, 0, 0.007],
                    ours: false
                },
                {
                    name: 'Kontext',
                    values: [0.05, 0.02, 0.048, 0.007, 0, 0.02, 0.01, 0, 0.019],
                    ours: false
                },
                {
                    name: 'Qwen-IE-2509',
                    values: [0.23, 0.04, 0.069, 0, 0, 0.02, 0.023, 0, 0.048],
                    ours: false
                },
                {
                    name: 'FlowInOne (Ours)',
                    values: [0.89, 0.7, 0.355, 0.727, 0.302, 0.52, 0.292, 0.535, 0.54],
                    ours: true
                }
            ]
        },
        {
            id: 'qwen',
            label: 'Qwen3.5',
            caption: 'Evaluator: Qwen3.5',
            rows: [
                {
                    name: 'Nano Banana',
                    values: [0.6, 0.959, 0.386, 0.367, 0.257, 0.04, 0.321, 0.744, 0.469],
                    ours: false
                },
                {
                    name: 'Omnigen2',
                    values: [0.03, 0.02, 0.017, 0.034, 0, 0, 0.003, 0.047, 0.019],
                    ours: false
                },
                {
                    name: 'Kontext',
                    values: [0.05, 0.02, 0.042, 0.133, 0, 0.06, 0.047, 0.093, 0.056],
                    ours: false
                },
                {
                    name: 'Qwen-IE-2509',
                    values: [0.27, 0.06, 0.08, 0.087, 0.047, 0.04, 0.033, 0.047, 0.083],
                    ours: false
                },
                {
                    name: 'FlowInOne (Ours)',
                    values: [0.859, 0.72, 0.354, 0.713, 0.272, 0.32, 0.306, 0.481, 0.503],
                    ours: true
                }
            ]
        },
        {
            id: 'human',
            label: 'Human',
            caption: 'Evaluator: Human judges',
            rows: [
                {
                    name: 'Nano Banana',
                    values: [0.602, 0.904, 0.271, 0.25, 0.2, 0.05, 0.229, 0.742, 0.406],
                    ours: false
                },
                {
                    name: 'Omnigen2',
                    values: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ours: false
                },
                {
                    name: 'Kontext',
                    values: [0, 0, 0.043, 0, 0, 0, 0, 0.1, 0.018],
                    ours: false
                },
                {
                    name: 'Qwen-IE-2509',
                    values: [0.067, 0, 0.029, 0, 0, 0, 0, 0, 0.012],
                    ours: false
                },
                {
                    name: 'FlowInOne (Ours)',
                    values: [0.8, 0.645, 0.242, 0.705, 0.255, 0.28, 0.255, 0.4, 0.449],
                    ours: true
                }
            ]
        }
    ]
};

function vpFormatRatio(v) {
    var s = Number(v).toFixed(3);
    if (s.charAt(0) === '0') return s.slice(1);
    return s;
}

function vpColumnMaxima(rows) {
    var n = rows[0].values.length;
    var maxes = [];
    var r;
    var c;
    for (c = 0; c < n; c++) {
        var m = -1;
        for (r = 0; r < rows.length; r++) {
            var v = rows[r].values[c];
            if (v > m) m = v;
        }
        maxes.push(m);
    }
    return maxes;
}

function vpIsColumnBest(v, colMax) {
    return Math.abs(v - colMax) < 0.00055;
}

function vpBuildTable(evalBlock) {
    var cols = VP_BENCH.cols;
    var maxes = vpColumnMaxima(evalBlock.rows);
    var ths = cols
        .map(function (c) {
            return '<th scope="col">' + c + '</th>';
        })
        .join('');
    var body = evalBlock.rows
        .map(function (row) {
            var trClass = row.ours ? ' class="vp-row--ours"' : '';
            var cells = row.values
                .map(function (v, i) {
                    var b = vpIsColumnBest(v, maxes[i]) ? ' vp-num--best' : '';
                    return '<td class="vp-num' + b + '">' + vpFormatRatio(v) + '</td>';
                })
                .join('');
            return (
                '<tr' +
                trClass +
                '><th scope="row">' +
                row.name +
                '</th>' +
                cells +
                '</tr>'
            );
        })
        .join('');
    return (
        '<div class="vp-table-wrap">' +
        '<table class="vp-results-table">' +
        '<colgroup><col class="vp-col-method"><col span="9" class="vp-col-metric"></colgroup>' +
        '<caption>' +
        evalBlock.caption +
        '</caption>' +
        '<thead><tr><th scope="col">Method</th>' +
        ths +
        '</tr></thead><tbody>' +
        body +
        '</tbody></table></div>'
    );
}

function initVpBenchResults() {
    var root = document.getElementById('vpBenchResultsRoot');
    if (!root || !VP_BENCH.evaluators.length) return;

    var firstId = VP_BENCH.evaluators[0].id;
    var tabsHtml =
        '<div class="vp-tablist" role="tablist" aria-label="VP-Bench evaluator">' +
        VP_BENCH.evaluators
            .map(function (ev, i) {
                var active = ev.id === firstId;
                return (
                    '<button type="button" class="vp-tab' +
                    (active ? ' is-active' : '') +
                    '" role="tab" id="vp-tab-' +
                    ev.id +
                    '" aria-selected="' +
                    (active ? 'true' : 'false') +
                    '" aria-controls="vp-panel-' +
                    ev.id +
                    '" data-vp-tab="' +
                    ev.id +
                    '">' +
                    ev.label +
                    '</button>'
                );
            })
            .join('') +
        '</div>';

    var panelsHtml = VP_BENCH.evaluators
        .map(function (ev, i) {
            var active = ev.id === firstId;
            return (
                '<div class="vp-panel" role="tabpanel" id="vp-panel-' +
                ev.id +
                '" aria-labelledby="vp-tab-' +
                ev.id +
                '"' +
                (active ? '' : ' hidden') +
                '>' +
                vpBuildTable(ev) +
                '</div>'
            );
        })
        .join('');

    root.innerHTML = tabsHtml + panelsHtml;

    var tabButtons = root.querySelectorAll('[data-vp-tab]');
    var panels = root.querySelectorAll('.vp-panel');

    function activate(id) {
        tabButtons.forEach(function (btn) {
            var on = btn.getAttribute('data-vp-tab') === id;
            btn.classList.toggle('is-active', on);
            btn.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        panels.forEach(function (panel) {
            var on = panel.id === 'vp-panel-' + id;
            if (on) panel.removeAttribute('hidden');
            else panel.setAttribute('hidden', '');
        });
    }

    tabButtons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            activate(btn.getAttribute('data-vp-tab'));
        });
    });
}

function initBenchFigureLightbox() {
    var lb = document.getElementById('benchLightbox');
    var imgEl = document.getElementById('benchLightboxImg');
    if (!lb || !imgEl) return;

    function openFrom(btn) {
        var src = btn.getAttribute('data-full-src');
        var alt = btn.getAttribute('data-full-alt') || '';
        if (!src) return;
        lb._benchReturnFocus = btn;
        imgEl.src = src;
        imgEl.alt = alt;
        lb.removeAttribute('hidden');
        document.body.classList.add('bench-lightbox-open');
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                lb.classList.add('is-open');
            });
        });
        window.setTimeout(function () {
            var closeBtn = lb.querySelector('.bench-lightbox-close');
            if (closeBtn) closeBtn.focus();
        }, 100);
    }

    document.querySelectorAll('.bench-pair-zoom').forEach(function (btn) {
        btn.addEventListener('click', function () {
            openFrom(btn);
        });
    });

    lb.querySelectorAll('[data-bench-lightbox-close]').forEach(function (el) {
        el.addEventListener('click', function (e) {
            e.preventDefault();
            closeBenchLightbox();
        });
    });
}

function initTeaserLightbox() {
    const trigger = document.getElementById('teaserLightboxTrigger');
    const lightbox = document.getElementById('teaserLightbox');
    if (!trigger || !lightbox) return;

    const backdrop = lightbox.querySelector('.teaser-lightbox-backdrop');
    const closeBtn = lightbox.querySelector('.teaser-lightbox-close');

    function closeLb() {
        lightbox.classList.remove('is-open');
        document.body.classList.remove('teaser-lightbox-open');
        trigger.setAttribute('aria-expanded', 'false');
        lightbox.addEventListener('transitionend', function onEnd(e) {
            if (e.target !== lightbox || e.propertyName !== 'opacity') return;
            lightbox.setAttribute('hidden', '');
            lightbox.removeEventListener('transitionend', onEnd);
            trigger.focus();
        });
    }

    function openLb() {
        lightbox.removeAttribute('hidden');
        document.body.classList.add('teaser-lightbox-open');
        trigger.setAttribute('aria-expanded', 'true');
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                lightbox.classList.add('is-open');
            });
        });
        window.setTimeout(function () {
            if (closeBtn) closeBtn.focus();
        }, 120);
    }

    trigger.addEventListener('click', function () {
        openLb();
    });
    if (backdrop) {
        backdrop.addEventListener('click', closeLb);
    }
    if (closeBtn) {
        closeBtn.addEventListener('click', closeLb);
    }
}

function initSectionQuickNav() {
    var nav = document.getElementById('sectionQuickNav');
    if (!nav || !window.IntersectionObserver) return;

    var links = [].slice.call(nav.querySelectorAll('.section-quick-nav__link'));
    if (!links.length) return;

    var sections = links
        .map(function (a) {
            var id = (a.getAttribute('href') || '').replace(/^#/, '');
            return id ? document.getElementById(id) : null;
        })
        .filter(Boolean);

    if (!sections.length) return;

    function setActiveById(activeId) {
        links.forEach(function (a) {
            var id = (a.getAttribute('href') || '').replace(/^#/, '');
            var on = id === activeId;
            a.classList.toggle('is-active', on);
            if (on) a.setAttribute('aria-current', 'location');
            else a.removeAttribute('aria-current');
        });
    }

    var io = new IntersectionObserver(
        function (entries) {
            var visible = entries.filter(function (e) {
                return e.isIntersecting;
            });
            if (!visible.length) return;
            visible.sort(function (a, b) {
                return b.intersectionRatio - a.intersectionRatio;
            });
            setActiveById(visible[0].target.id);
        },
        {
            root: null,
            rootMargin: '-10% 0px -52% 0px',
            threshold: [0, 0.08, 0.2, 0.35, 0.5, 0.75, 1]
        }
    );

    sections.forEach(function (sec) {
        io.observe(sec);
    });

    function syncFromScroll() {
        if (window.scrollY < 120) {
            setActiveById('intro');
        }
    }

    window.addEventListener(
        'scroll',
        function () {
            syncFromScroll();
        },
        { passive: true }
    );
    syncFromScroll();
}

document.addEventListener('DOMContentLoaded', function () {
    initQualResultsGrid();
    initQualCompareLightbox();
    initBenchFigureLightbox();
    initVpBenchResults();
    initTeaserLightbox();
    initSectionQuickNav();
});