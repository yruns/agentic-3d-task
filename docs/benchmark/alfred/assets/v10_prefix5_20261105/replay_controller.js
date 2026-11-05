
(() => {
  const root = document.getElementById('alfred-replays');
  const episodes = JSON.parse(document.getElementById('alfred-replay-data').textContent);
  const select = root.querySelector('#alfred-case');
  const slider = root.querySelector('#alfred-step');
  const state = root.querySelector('#alfred-state');
  const label = root.querySelector('#alfred-step-label');
  const rgb = root.querySelector('#alfred-rgb');
  const bev = root.querySelector('#alfred-bev');
  const command = root.querySelector('#alfred-command');
  const commandLabel = root.querySelector('#alfred-command-label');
  const feedback = root.querySelector('#alfred-feedback');
  const back = root.querySelector('#alfred-back');
  const next = root.querySelector('#alfred-next');
  const play = root.querySelector('#alfred-play');
  const plot = root.querySelector('#alfred-trajectory');
  let timer = null;
  let episodeIndex = 0;
  let step = 0;
  function selected() { return episodes[episodeIndex].snapshots[step]; }
  function save() {
    if (window.openai && typeof window.openai.setWidgetState === 'function') {
      window.openai.setWidgetState({
        modelContent: { episode: episodes[episodeIndex].id, frame: selected().frame, decision: selected().decision },
        privateContent: { episode: episodes[episodeIndex].id, step }
      }).catch(() => {});
    }
  }
  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null; play.textContent = '播放关键帧'; play.setAttribute('aria-pressed', 'false');
  }
  function drawTrajectory() {
    const points = episodes[episodeIndex].trajectory;
    const width = Math.max(280, Math.round(plot.getBoundingClientRect().width || 680));
    const height = 150, left = 64, right = width - 24, top = 12, bottom = 108;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    let xmin = Math.min(...xs) - .25, xmax = Math.max(...xs) + .25;
    let ymin = Math.min(...ys) - .25, ymax = Math.max(...ys) + .25;
    const scale = Math.min((right-left)/(xmax-xmin), (bottom-top)/(ymax-ymin));
    const cx = (xmin+xmax)/2, cy = (ymin+ymax)/2;
    xmin = cx-(right-left)/(2*scale); xmax = cx+(right-left)/(2*scale);
    ymin = cy-(bottom-top)/(2*scale); ymax = cy+(bottom-top)/(2*scale);
    const px = x => left + (x - xmin) / (xmax - xmin) * (right - left);
    const py = y => bottom - (y - ymin) / (ymax - ymin) * (bottom - top);
    const path = values => values.map(p => `${px(p.x).toFixed(2)},${py(p.y).toFixed(2)}`).join(' ');
    const past = points.filter(p => p.frame <= selected().frame);
    const current = past[past.length - 1];
    plot.setAttribute('viewBox', `0 0 ${width} ${height}`);
    plot.innerHTML = `<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="none" stroke="var(--border)"/>
      <polyline points="${path(points)}" fill="none" stroke="var(--border)" stroke-width="2"/>
      <polyline points="${path(past)}" fill="none" stroke="var(--viz-series-1)" stroke-width="2"/>
      <circle cx="${px(current.x)}" cy="${py(current.y)}" r="5" fill="var(--viz-series-1)"/>
      <text x="${left}" y="126">${xmin.toFixed(1)}</text><text x="${right}" y="126" text-anchor="end">${xmax.toFixed(1)}</text>
      <text x="54" y="${bottom}" text-anchor="end">${ymin.toFixed(1)}</text><text x="54" y="${top+10}" text-anchor="end">${ymax.toFixed(1)}</text>
      <text x="${(left+right)/2}" y="145" text-anchor="middle" class="axis-title" data-axis="x">X (m)</text>
      <text x="10" y="64" class="axis-title" data-axis="y">Y (m)</text>`;
  }
  function render() {
    const episode = episodes[episodeIndex], frame = selected();
    select.value = episode.id; slider.max = episode.snapshots.length - 1; slider.value = step;
    state.textContent = `${frame.label} · 观测帧 ${frame.frame} · 对象节点 ${frame.nodes} · 持有 ${frame.held === null ? '无' : frame.held}`;
    label.textContent = `关键帧 ${step + 1} / ${episode.snapshots.length}`;
    rgb.src = frame.rgb; bev.src = frame.bev;
    commandLabel.textContent = frame.decision === null ? '结果：' : '下一决策：';
    command.textContent = frame.action; feedback.textContent = frame.feedback;
    back.disabled = step === 0; next.disabled = step === episode.snapshots.length - 1;
    drawTrajectory();
  }
  function change(value) { step = Math.max(0, Math.min(episodes[episodeIndex].snapshots.length - 1, value)); render(); }
  function restore(saved) {
    const value = saved && saved.privateContent;
    if (!value || typeof value.episode !== 'string') return;
    const index = episodes.findIndex(e => e.id === value.episode);
    if (index < 0 || !Number.isInteger(value.step)) return;
    stop(); episodeIndex = index; change(value.step);
  }
  select.addEventListener('change', () => { stop(); episodeIndex = episodes.findIndex(e => e.id === select.value); change(0); save(); });
  slider.addEventListener('input', () => { stop(); change(Number(slider.value)); save(); });
  back.addEventListener('click', () => { stop(); change(step - 1); save(); });
  next.addEventListener('click', () => { stop(); change(step + 1); save(); });
  play.addEventListener('click', () => {
    if (timer !== null) { stop(); save(); return; }
    if (step === episodes[episodeIndex].snapshots.length - 1) change(0);
    play.textContent = '暂停'; play.setAttribute('aria-pressed', 'true');
    timer = window.setInterval(() => { change(step + 1); if (step === episodes[episodeIndex].snapshots.length - 1) { stop(); save(); } }, 1500);
    save();
  });
  window.addEventListener('openai:set_globals', event => { if (event.detail && event.detail.globals) restore(event.detail.globals.widgetState); });
  new ResizeObserver(drawTrajectory).observe(plot);
  render();
  if (window.openai) restore(window.openai.widgetState);
})();
