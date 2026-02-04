document.addEventListener('DOMContentLoaded', () => {
  const codeBlocks = document.querySelectorAll('.highlight');

  codeBlocks.forEach(block => {
    const header = document.createElement('div');
    header.className = 'code-header';

    const controls = document.createElement('div');
    controls.className = 'window-controls';
    ['red', 'yellow', 'green'].forEach(color => {
      const dot = document.createElement('span');
      dot.className = `window-dot ${color}`;
      controls.appendChild(dot);
    });
    header.appendChild(controls);

    let langName = '';
    const code = block.querySelector('code[data-lang]');
    if (code) {
      langName = code.getAttribute('data-lang');
    } else {
      const codeClass = block.querySelector('code');
      if (codeClass) {
        const match = codeClass.className.match(/language-(\w+)/);
        if (match) langName = match[1];
      }
    }

    if (langName) {
      const langLabel = document.createElement('span');
      langLabel.className = 'lang-label';
      langLabel.textContent = langName;
      header.appendChild(langLabel);
    }

    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-btn';
    copyBtn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span>Copy</span>
    `;
    
    copyBtn.addEventListener('click', async () => {
      let codeText = '';
      const table = block.querySelector('table');
      if (table) {
        const codeTd = table.querySelector('td:last-child');
        if (codeTd) codeText = codeTd.innerText;
      } else {
        codeText = block.querySelector('code').innerText;
      }

      try {
        await navigator.clipboard.writeText(codeText);
        copyBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-green-500">
            <polyline points="20 6 9 17 4 12"></polyline>
          </svg>
          <span>Copied!</span>
        `;
        setTimeout(() => {
          copyBtn.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          `;
        }, 2000);
      } catch (err) {
        console.error('Failed to copy!', err);
      }
    });

    header.appendChild(copyBtn);
    
    const wrapper = document.createElement('div');
    wrapper.className = 'code-block-wrapper';
    
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(header);
    wrapper.appendChild(block);
  });
});
