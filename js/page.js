// Page chrome: equations, scroll reveals, BibTeX copy.
(() => {
  'use strict';

  document.querySelectorAll('.eq[data-tex]').forEach((el) => {
    try {
      window.katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false });
    } catch (err) {
      console.error('KaTeX:', err);
      el.textContent = el.dataset.tex;
    }
  });

  const io = new IntersectionObserver((entries) => entries.forEach((e) => {
    if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }), { rootMargin: '0px 0px -8% 0px' });
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

  const copy = document.getElementById('copy');
  copy.addEventListener('click', async () => {
    const text = document.getElementById('bibtex').textContent;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = 'Copied';
    } catch (err) {
      console.error('Clipboard:', err);
      copy.textContent = 'Select & copy';
    }
    setTimeout(() => { copy.textContent = 'Copy'; }, 1600);
  });
})();
