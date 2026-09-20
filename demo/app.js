'use strict';

const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const state = { data: null, dataset: null, model: 'trained', rows: [], index: 0 };
const measured = (value) => typeof value === 'number' && Number.isFinite(value);
const pct = (value) => measured(value) ? `${(value * 100).toFixed(2)}%` : 'Not measured';
const num = (value) => measured(value) ? value.toFixed(3) : 'Not measured';
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function svgNode(tag, attrs = {}, text) {
  const element = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  return element;
}
function chart(target, title, description, width = 480, height = 300) {
  const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-labelledby': `${target}-title ${target}-desc` });
  svg.append(svgNode('title', { id: `${target}-title` }, title), svgNode('desc', { id: `${target}-desc` }, description));
  $(target).replaceChildren(svg);
  return svg;
}
function line(svg, x1, y1, x2, y2, dashed = false) {
  svg.append(svgNode('line', { x1, y1, x2, y2, stroke: '#405345', 'stroke-dasharray': dashed ? '5 5' : 'none' }));
}
function text(svg, x, y, value, attrs = {}) {
  svg.append(svgNode('text', { x, y, ...attrs }, value));
}
function table(target, caption, headers, rows) {
  const element = node('table');
  element.append(node('caption', caption));
  const head = node('thead');
  const headRow = node('tr');
  headers.forEach((label) => { const cell = node('th', label); cell.scope = 'col'; headRow.append(cell); });
  head.append(headRow);
  const body = node('tbody');
  rows.forEach((values) => {
    const row = node('tr');
    values.forEach((value, i) => { const cell = node(i === 0 ? 'th' : 'td', String(value)); if (i === 0) cell.scope = 'row'; row.append(cell); });
    body.append(row);
  });
  element.append(head, body);
  $(target).replaceChildren(element);
}
function bar(label, value, frozen = false) {
  const row = node('div', undefined, 'bar-row');
  const heading = node('div', undefined, 'bar-label');
  heading.append(node('span', label), node('strong', pct(value)));
  const track = node('div', undefined, 'bar-track');
  track.setAttribute('aria-hidden', 'true');
  const fill = node('div', undefined, `bar-fill${frozen ? ' frozen' : ''}`);
  fill.style.width = `${measured(value) ? Math.max(0, Math.min(1, value)) * 100 : 0}%`;
  track.append(fill); row.append(heading, track);
  return row;
}
function renderTabs() {
  state.data.datasets.forEach((dataset, index) => {
    const tab = node('button');
    tab.type = 'button'; tab.id = `tab-${dataset.id}`;
    tab.setAttribute('role', 'tab'); tab.setAttribute('aria-controls', 'experiment');
    tab.append(node('span', `EXPERIMENT / 0${index + 1}`, 'tab-number'), node('span', dataset.label));
    tab.addEventListener('click', () => selectDataset(dataset));
    tab.addEventListener('keydown', (event) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'Home', 'End'];
      if (!keys.includes(event.key)) return;
      event.preventDefault();
      const count = state.data.datasets.length;
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? count - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + count) % count;
      const chosen = state.data.datasets[next];
      selectDataset(chosen); $(`tab-${chosen.id}`).focus();
    });
    $('dataset-tabs').append(tab);
  });
}
function selectDataset(dataset) {
  state.dataset = dataset;
  state.index = 0;
  const previous = state.model;
  const supported = dataset.models.some((model) => model.id === previous);
  if (!supported) state.model = dataset.models.find((model) => model.id === 'trained')?.id || dataset.models[0].id;
  state.data.datasets.forEach((item) => {
    const active = item.id === dataset.id;
    const tab = $(`tab-${item.id}`); tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1;
  });
  $('experiment').setAttribute('aria-labelledby', `tab-${dataset.id}`);
  $('dataset-title').textContent = dataset.title;
  $('dataset-scope').textContent = dataset.subtitle;
  $('report-link').href = dataset.report_url;
  const special = {
    banking: 'Selected 10 of 77 intents—not the complete BANKING77 benchmark.',
    clinc: '10 supported intents plus the official out-of-scope population. Overall accuracy is dominated by OOS records; inspect conditional metrics below.',
    contractnli: '10 complete short documents / 170 decisions. Length filtering excludes longer documents; this is not full ContractNLI or evidence extraction. One extra correct decision is not a robust gain.',
    bitcoin: 'No accuracy improvement over Frozen or the train-frequency baseline. Only 32 exploratory observations; the test set was reused. Not forecasting evidence or a trading system.'
  };
  $('scope-note').textContent = special[dataset.id] || '';
  $('scope-note').className = `scope-note${dataset.id === 'bitcoin' ? ' warning' : ''}`;
  $('model-select').replaceChildren(...dataset.models.map((model) => {
    const option = node('option', model.label); option.value = model.id; return option;
  }));
  $('model-select').value = state.model;
  $('model-notice').textContent = !supported ? `The previous model is not available for ${dataset.label}; selected ${dataset.models.find((model) => model.id === state.model).label}. Frozen + calibration was not recorded for Bitcoin.` : dataset.id === 'bitcoin' ? 'Frozen + calibration was not recorded for Bitcoin. Only available models are listed.' : 'Only recorded model variants are listed. Calibration uses a separate calibration split.';
  renderOverview(); renderProvenance(); renderModel();
}
function renderOverview() {
  const d = state.dataset;
  const base = d.models.find((m) => m.id === 'base');
  const trained = d.models.find((m) => m.id === 'trained');
  $('accuracy-bars').replaceChildren(bar('Frozen', base?.accuracy, true), bar('LoRA', trained?.accuracy));
  const delta = measured(base?.accuracy) && measured(trained?.accuracy) ? (trained.accuracy - base.accuracy) * 100 : null;
  $('accuracy-delta').textContent = delta === null ? 'Not measured' : `${delta > 0 ? '+' : ''}${delta.toFixed(2)} pp`;
  $('accuracy-delta').className = delta < 0 ? 'negative' : delta > 0 ? 'positive' : '';
  $('baselines').textContent = `Train-frequency baseline ${pct(d.prior_accuracy)} · TF–IDF ${pct(d.tfidf_accuracy)}`;
  $('step-count').textContent = `${d.training_steps} steps`;
  const history = d.history.filter((point) => measured(point.step) && measured(point.loss));
  const svg = chart('loss-chart', 'Recorded training loss', `Unsmoothed batch loss over ${d.training_steps} training steps.`, 480, 150);
  if (history.length) {
    const maximum = Math.max(...history.map((point) => point.loss), 0.001);
    const x = (step) => 45 + step / Math.max(d.training_steps, 1) * 415;
    const y = (loss) => 115 - loss / maximum * 100;
    line(svg, 45, 15, 45, 115); line(svg, 45, 115, 460, 115);
    text(svg, 36, 20, maximum.toFixed(2), { 'text-anchor': 'end' }); text(svg, 36, 118, '0', { 'text-anchor': 'end' });
    text(svg, 45, 137, '0'); text(svg, 250, 137, 'Training step', { 'text-anchor': 'middle' }); text(svg, 460, 137, d.training_steps, { 'text-anchor': 'end' });
    svg.append(svgNode('polyline', { points: history.map((p) => `${x(p.step)},${y(p.loss)}`).join(' '), fill: 'none', stroke: '#cbef8c', 'stroke-width': 1.6 }));
    history.forEach((point) => { const circle = svgNode('circle', { cx: x(point.step), cy: y(point.loss), r: 2, fill: '#cbef8c' }); circle.append(svgNode('title', {}, `Step ${point.step}: loss ${num(point.loss)}`)); svg.append(circle); });
  } else text(svg, 45, 70, 'Training history not measured');
  $('validation').textContent = `Validation loss: ${num(d.validation_before)} → ${num(d.validation_after)} · ${d.validation_improved === true ? 'improved' : d.validation_improved === false ? 'did not improve' : 'not measured'}. LoRA calibration temperature: ${num(d.temperature)}.`;
  $('validation').className = `fine${d.validation_improved === false ? ' negative' : ''}`;
}
function renderModel() {
  const model = state.dataset.models.find((item) => item.id === state.model);
  const metrics = [['Accuracy', model.accuracy, pct, 'Higher is better'], ['Macro F1', model.macro_f1, pct, 'Equal weight per class'], ['NLL', model.nll, num, 'Lower is better'], ['ECE · 10 bins', model.ece, pct, 'Lower is better']];
  $('metric-cards').replaceChildren(...metrics.map(([label, value, format, hint]) => {
    const card = node('div', undefined, 'metric-card'); card.append(node('span', label, 'fine'), node('strong', format(value)), node('span', hint, 'fine')); return card;
  }));
  const taskItems = [node('span', `Brier score ${num(model.brier)} · lower is better`, 'task-item')];
  if (state.dataset.id === 'clinc') {
    [['OOS recall', 'out_of_scope_recall'], ['In-scope false rejection', 'in_scope_false_rejection_rate'], ['Supported-intent accuracy', 'supported_intent_accuracy']].forEach(([label, key]) => {
      const item = node('span', label, 'task-item'); item.append(node('strong', pct(model.task_metrics?.[key]))); taskItems.push(item);
    });
  }
  $('task-metrics').replaceChildren(...taskItems);
  renderReliability(model); renderConfusion(model); filterRecords();
}
function renderReliability(model) {
  const bins = model.reliability.filter((bin) => bin.count > 0 && measured(bin.accuracy) && measured(bin.mean_confidence));
  const svg = chart('reliability-chart', `${model.label}: reliability`, 'Observed accuracy against mean confidence, both from 0 to 100 percent. The accessible table contains every nonempty bin and its record count.');
  const x = (value) => 55 + value * 390;
  const y = (value) => 245 - value * 210;
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    line(svg, 55, y(tick), 445, y(tick)); text(svg, 45, y(tick) + 4, `${tick * 100}%`, { 'text-anchor': 'end' }); text(svg, x(tick), 265, `${tick * 100}%`, { 'text-anchor': 'middle' });
  });
  text(svg, 55, 18, 'Observed accuracy'); text(svg, 250, 294, 'Mean confidence', { 'text-anchor': 'middle' });
  line(svg, x(0), y(0), x(1), y(1), true);
  bins.forEach((bin) => {
    const circle = svgNode('circle', { cx: x(bin.mean_confidence), cy: y(bin.accuracy), r: 5, fill: '#cbef8c', stroke: '#101813', 'stroke-width': 1.5 });
    circle.append(svgNode('title', {}, `${pct(bin.lower)}–${pct(bin.upper)} bin: ${bin.count} records; confidence ${pct(bin.mean_confidence)}; accuracy ${pct(bin.accuracy)}`)); svg.append(circle);
  });
  table('reliability-table', `${model.label}: nonempty confidence bins`, ['Bin', 'Count', 'Confidence', 'Accuracy'], bins.map((bin) => [`${pct(bin.lower)}–${pct(bin.upper)}`, bin.count, pct(bin.mean_confidence), pct(bin.accuracy)]));
}
function renderConfusion(model) {
  const labels = state.dataset.choices;
  const n = labels.length;
  const svg = chart('confusion-chart', `${model.label}: confusion matrix`, 'Rows are true labels; columns are predicted labels. Numbered labels map to the full label names in the table below. Each cell title names both labels and its count.');
  const size = Math.min(210 / n, 55);
  const width = n * size;
  const start = (480 - width) / 2;
  const max = Math.max(...model.confusion.flat(), 1);
  text(svg, 240, 18, 'Predicted label →', { 'text-anchor': 'middle' });
  text(svg, 18, 155, 'True label →', { transform: 'rotate(-90 18 155)', 'text-anchor': 'middle' });
  labels.forEach((label, i) => {
    text(svg, start + (i + 0.5) * size, 39, i + 1, { 'text-anchor': 'middle' });
    text(svg, start - 10, 48 + (i + 0.5) * size + 4, i + 1, { 'text-anchor': 'end' });
    model.confusion[i].forEach((count, j) => {
      const rect = svgNode('rect', { x: start + j * size, y: 48 + i * size, width: size - 1, height: size - 1, rx: 2, fill: '#cbef8c', 'fill-opacity': 0.06 + count / max * 0.94 });
      rect.append(svgNode('title', {}, `True: ${label}; predicted: ${labels[j]}; count: ${count}`)); svg.append(rect);
      if (n <= 3) text(svg, start + (j + 0.5) * size, 52 + (i + 0.5) * size, count, { 'text-anchor': 'middle', style: `fill:${count / max > 0.5 ? '#101813' : '#f3f0e6'}` });
    });
  });
  text(svg, 240, 284, 'Full label names and exact counts in the table ↓', { 'text-anchor': 'middle', style: 'font-size:11px' });
  table('confusion-table', 'True labels (rows) × predicted labels (columns). Numbers match the chart.', ['True / Predicted', ...labels.map((label, i) => `${i + 1}. ${label}`)], labels.map((label, i) => [`${i + 1}. ${label}`, ...model.confusion[i]]));
}
function filterRecords() {
  const previousId = state.rows[state.index]?.id;
  state.rows = state.dataset.rows.filter((row) => !$('errors-only').checked || row.models[state.model].answer !== row.label);
  state.index = Math.max(0, state.rows.findIndex((row) => row.id === previousId));
  $('record-select').replaceChildren(...state.rows.map((row, index) => { const option = node('option', `${index + 1}. ${row.id}`); option.value = String(index); return option; }));
  renderRecord();
}
function renderRecord() {
  const row = state.rows[state.index];
  $('record-select').disabled = !row;
  $('record-select').value = String(state.index);
  $('record-prev').disabled = !row || state.index === 0;
  $('record-next').disabled = !row || state.index === state.rows.length - 1;
  $('record-position').textContent = row ? `${state.index + 1} of ${state.rows.length} ${$('errors-only').checked ? 'errors' : 'held-out records'} · ${state.dataset.test_count} total held-out records` : 'No errors for this model. Turn off Errors only to inspect all held-out records.';
  $('record-detail').replaceChildren(); $('probability-bars').replaceChildren();
  if (!row) return;
  const prediction = row.models[state.model];
  [['Record ID', row.id], ['Ground truth', row.label], [prediction.answer === row.label ? 'Prediction · correct' : 'Prediction · incorrect', prediction.answer]].forEach(([label, value], i) => {
    const cell = node('div', undefined, 'record-value'); cell.append(node('span', label, 'record-label'), node('span', value, i === 2 ? prediction.answer === row.label ? 'positive' : 'negative' : '')); $('record-detail').append(cell);
  });
  $('probability-bars').replaceChildren(...state.dataset.choices.map((choice, index) => bar(choice, prediction.probabilities[index], state.model.startsWith('base'))));
}
function renderProvenance() {
  const d = state.dataset;
  $('notes').replaceChildren(...d.notes.map((note) => node('li', note)));
  const coverage = $('coverage'); coverage.replaceChildren();
  coverage.append(node('p', `Recorded splits: ${Object.entries(d.counts).map(([split, count]) => `${split}: ${count}`).join(' · ')}.`, 'fine'));
  Object.entries(d.coverage).forEach(([split, item]) => {
    coverage.append(node('p', `${split}: ${item.retained_groups} of ${item.selected_groups} source groups retained after deduplication and length filtering (${pct(item.group_retention_fraction)}); ${item.token_excluded_groups} groups excluded by token length. These are eligible pools before demo-budget selection, not final evaluated counts.`, 'fine'));
  });
  if (!Object.keys(d.coverage).length) coverage.append(node('p', 'Source-group filtering coverage was not recorded for this experiment.', 'fine'));
  const hashes = $('source-hashes'); hashes.replaceChildren();
  Object.entries(state.data.source_sha256).filter(([path]) => path.startsWith(`results/${d.id}/`)).forEach(([path, hash]) => {
    hashes.append(node('dt', path), node('dd', hash));
  });
}
$('model-select').addEventListener('change', (event) => { state.model = event.target.value; renderModel(); });
$('errors-only').addEventListener('change', filterRecords);
$('record-select').addEventListener('change', (event) => { state.index = Number(event.target.value); renderRecord(); });
$('record-prev').addEventListener('click', () => { if (state.index > 0) { state.index--; renderRecord(); } });
$('record-next').addEventListener('click', () => { if (state.index + 1 < state.rows.length) { state.index++; renderRecord(); } });
async function load() {
  try {
    if (location.protocol === 'file:') throw new Error('This dashboard needs HTTP to read data.json. From the repository root run: python3 -m http.server 8765 --bind 127.0.0.1, then open http://127.0.0.1:8765/demo/.');
    const response = await fetch('data.json');
    if (!response.ok) throw new Error(`data.json returned HTTP ${response.status}. Serve the repository root with python3 -m http.server 8765 --bind 127.0.0.1 and open /demo/. Confirm demo/data.json exists.`);
    const data = await response.json();
    if (data.format !== 'ifllm-learn.visual-demo.v1' || !Array.isArray(data.datasets) || !data.datasets.length) throw new Error('Unsupported experiment data. Restore demo/data.json from this repository and reload.');
    state.data = data;
    renderTabs(); selectDataset(data.datasets[0]);
    $('load-status').textContent = ''; $('lab').hidden = false;
  } catch (error) {
    $('lab').hidden = true;
    $('load-status').setAttribute('role', 'alert');
    $('load-status').className = 'warning';
    $('load-status').textContent = `Unable to load recorded experiments. ${error.message} If the server is already running, confirm demo/data.json is present and reload this page.`;
  }
}
load();
