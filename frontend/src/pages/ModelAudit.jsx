import { useMemo, useState } from 'react';
import { percent, trainingAudit } from '../data/trainingResults';

const METRIC_HELP = {
  validation: 'Mean macro-F1 across folds where one entire source dataset was excluded from training.',
  test: 'Mean macro-F1 across four held-out test collections. These collections differ in size and label balance.',
  brier: 'Probability error. Lower is better; zero would be perfect probability calibration.',
};

function ScoreBar({ value, colour, subtle = false }) {
  return <div className={`audit-score-track ${subtle ? 'subtle' : ''}`} aria-label={percent(value)}><i style={{ width: percent(value), background: colour }} /></div>;
}

function ModelLegend({ models }) {
  return <div className="audit-legend">{models.map(model => <span key={model.id}><i style={{ background: model.colour }} />{model.shortName}</span>)}</div>;
}

const PERFORMANCE_STAGES = [
  ['validationMeanF1', 'Source-held validation'],
  ['testMeanF1', 'Held-out tests'],
  ['contemporaryF1', '2026 diagnostic'],
];

function PerformanceGraph({ models }) {
  const x = index => 90 + index * 265;
  const y = value => 244 - value * 190;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return <section className="panel audit-panel audit-graph-card">
    <div className="audit-panel-head"><div><span className="eyebrow">GENERALIZATION CURVE</span><h3>Performance across evaluation settings</h3></div></div>
    <p className="audit-caption">Macro-F1 falls when the source or time period becomes less familiar. Lines connect evaluation settings for comparison; they are not a time-series forecast.</p>
    <div className="audit-svg-wrap">
      <svg className="performance-graph" viewBox="0 0 640 285" role="img" aria-label="Macro F1 across source-held validation, held-out tests, and contemporary diagnostic">
        {ticks.map(tick => <g key={tick}><line x1="58" y1={y(tick)} x2="620" y2={y(tick)} className="chart-grid" /><text x="49" y={y(tick) + 4} textAnchor="end" className="chart-axis-label">{Math.round(tick * 100)}%</text></g>)}
        {PERFORMANCE_STAGES.map(([, label], index) => <text key={label} x={x(index)} y="275" textAnchor="middle" className="chart-stage-label">{label}</text>)}
        {models.map(model => {
          const points = PERFORMANCE_STAGES.map(([key], index) => `${x(index)},${y(model[key])}`).join(' ');
          return <g key={model.id}>
            <polyline points={points} fill="none" stroke={model.colour} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
            {PERFORMANCE_STAGES.map(([key], index) => <g key={key}><circle cx={x(index)} cy={y(model[key])} r="6" fill={model.colour} stroke="var(--paper)" strokeWidth="3" /><text x={x(index)} y={y(model[key]) - 12} textAnchor="middle" className="chart-value-label" fill={model.colour}>{percent(model[key])}</text></g>)}
          </g>;
        })}
      </svg>
    </div>
    <ModelLegend models={models} />
  </section>;
}

function DatasetGraph({ models }) {
  return <section className="panel audit-panel audit-graph-card">
    <div className="audit-panel-head"><div><span className="eyebrow">DATASET COMPARISON</span><h3>Held-out macro-F1</h3></div></div>
    <p className="audit-caption">Each collection is shown independently so strong ISOT performance cannot hide weaker BharatGather transfer.</p>
    <div className="dataset-graph">
      <div className="dataset-axis"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      {trainingAudit.datasetResults.map(row => <div className="dataset-graph-row" key={row.dataset}>
        <strong>{row.dataset}</strong>
        <div className="dataset-bars">{models.map(model => <div className="dataset-bar-line" key={model.id} title={`${model.shortName}: ${percent(row[model.id])}`}><i style={{ width: percent(row[model.id]), background: model.colour }} /><b>{percent(row[model.id])}</b></div>)}</div>
      </div>)}
    </div>
    <ModelLegend models={models} />
  </section>;
}

function ValidationChart({ models }) {
  return <section className="panel audit-panel audit-validation">
    <div className="audit-panel-head"><div><span className="eyebrow">ROBUSTNESS FIRST</span><h3>Unseen-source validation</h3></div><span className="audit-info" title={METRIC_HELP.validation}>?</span></div>
    <p className="audit-caption">This is the primary selection view. Lower scores here expose collection and publisher shift hidden by ordinary test splits.</p>
    <div className="audit-bar-groups">
      {models.map(model => <div className="audit-bar-group" key={model.id}>
        <div className="audit-bar-label"><strong>{model.shortName}</strong>{model.selected && <span>Selected</span>}</div>
        <div className="audit-bar-line"><small>Mean</small><ScoreBar value={model.validationMeanF1} colour={model.colour} /><b>{percent(model.validationMeanF1)}</b></div>
        <div className="audit-bar-line"><small>Worst</small><ScoreBar value={model.validationWorstF1} colour={model.colour} subtle /><b>{percent(model.validationWorstF1)}</b></div>
      </div>)}
    </div>
  </section>;
}

function DatasetHeatmap({ models }) {
  return <section className="panel audit-panel audit-datasets">
    <div className="audit-panel-head"><div><span className="eyebrow">HELD-OUT TESTS</span><h3>Macro-F1 by dataset</h3></div><span className="audit-info" title={METRIC_HELP.test}>?</span></div>
    <p className="audit-caption">Each cell gives equal importance to the real and fake classes. Darker cells are stronger.</p>
    <div className="audit-heatmap" role="table" aria-label="Held-out macro F1 by dataset and model" style={{ '--model-count': models.length }}>
      <div className="heatmap-row heatmap-head" role="row"><span>Dataset</span>{models.map(model => <span key={model.id}>{model.shortName.replace('Logistic ', 'Log. ')}</span>)}</div>
      {trainingAudit.datasetResults.map(row => <div className="heatmap-row" role="row" key={row.dataset}>
        <strong>{row.dataset}</strong>
        {models.map(model => <span className="heatmap-cell" key={model.id} style={{ '--heat': row[model.id] }}><b>{percent(row[model.id])}</b><i style={{ width: percent(row[model.id]), background: model.colour }} /></span>)}
      </div>)}
    </div>
  </section>;
}

function Thresholds({ models }) {
  return <section className="panel audit-panel audit-thresholds">
    <div className="audit-panel-head"><div><span className="eyebrow">OPERATING POINTS</span><h3>Validation-selected thresholds</h3></div></div>
    <p className="audit-caption">Scores at or above the marker become “fake”. These cutoffs are model-specific, not universal confidence levels.</p>
    <div className="threshold-list">{models.map(model => <div className="threshold-row" key={model.id}>
      <div><strong>{model.shortName}</strong><small>{model.type}</small></div>
      <div className="threshold-scale"><span>0</span><div><i style={{ left: percent(model.threshold), background: model.colour }} /><b style={{ left: percent(model.threshold), color: model.colour }}>{model.threshold.toFixed(2)}</b></div><span>1</span></div>
    </div>)}</div>
    <div className="audit-note"><b>Why they differ</b><p>Each model distributes probability differently. A shared 0.50 or 0.60 cutoff would change the false-positive/false-negative balance and was not the validation-selected operating point.</p></div>
  </section>;
}

function CalibrationPanel({ models }) {
  return <section className="panel audit-panel">
    <div className="audit-panel-head"><div><span className="eyebrow">PROBABILITY QUALITY</span><h3>Ranking & calibration</h3></div><span className="audit-info" title={METRIC_HELP.brier}>?</span></div>
    <div className="calibration-grid">{models.map(model => <article key={model.id}>
      <span style={{ color: model.colour }}>{model.shortName}</span>
      <div><small>Mean ROC-AUC</small><strong>{model.meanRocAuc.toFixed(4)}</strong></div>
      <div><small>Mean Brier ↓</small><strong>{model.meanBrier.toFixed(4)}</strong></div>
      <ScoreBar value={1 - model.meanBrier} colour={model.colour} />
    </article>)}</div>
    <p className="audit-footnote">High ROC-AUC means good ranking; it does not guarantee that the selected cutoff or displayed probability generalizes to a new source.</p>
  </section>;
}

function ConfusionMatrix({ model }) {
  const [[tn, fp], [fn, tp]] = model.confusion;
  return <article className="confusion-card">
    <div className="confusion-title"><i style={{ background: model.colour }} /><strong>{model.shortName}</strong><b>{percent(model.contemporaryF1)} F1</b></div>
    <div className="matrix-axis"><span>Predicted real</span><span>Predicted fake</span></div>
    <div className="matrix-grid"><div className="matrix-good"><b>{tn}</b><small>True real</small></div><div className="matrix-bad"><b>{fp}</b><small>False alarm</small></div><div className="matrix-bad"><b>{fn}</b><small>Missed fake</small></div><div className="matrix-good"><b>{tp}</b><small>Detected fake</small></div></div>
  </article>;
}

function ErrorTradeoff({ models }) {
  const maximum = Math.max(...models.map(model => model.errors.falsePositive + model.errors.falseNegative));
  return <section className="panel audit-panel audit-errors">
    <div className="audit-panel-head"><div><span className="eyebrow">ERROR ANALYSIS</span><h3>False alarms vs missed fake items</h3></div></div>
    <div className="error-legend"><span><i className="fp" />False positives</span><span><i className="fn" />False negatives</span></div>
    <div className="error-rows">{models.map(model => {
      const total = model.errors.falsePositive + model.errors.falseNegative;
      return <div className="error-row" key={model.id}><div><strong>{model.shortName}</strong><small>{total.toLocaleString()} total errors</small></div><div className="error-track" style={{ width: `${Math.max(28, total / maximum * 100)}%` }}><span className="fp" style={{ width: `${model.errors.falsePositive / total * 100}%` }}>{model.errors.falsePositive}</span><span className="fn" style={{ width: `${model.errors.falseNegative / total * 100}%` }}>{model.errors.falseNegative}</span></div></div>;
    })}</div>
    <div className="audit-note amber"><b>Forest trade-off</b><p>Random Forest catches more fake-labelled records, but its selected 0.37 cutoff produces 1,022 false alarms. Logistic Regression has the most balanced error profile.</p></div>
  </section>;
}

function ForestTuning() {
  return <section className="panel audit-panel audit-tuning">
    <div className="audit-panel-head"><div><span className="eyebrow">RANDOM FOREST</span><h3>Validation-only tuning audit</h3></div><span className="small-tag">No test selection</span></div>
    <div className="tuning-table"><div className="tuning-row tuning-head"><span>Configuration</span><span>Mean F1</span><span>Worst F1</span><span>Cutoff</span></div>{trainingAudit.randomForestTuning.map(row => <div className={`tuning-row ${row.selected ? 'selected' : ''}`} key={row.name}><span>{row.name}{row.selected && <b>Retained</b>}</span><strong>{row.mean.toFixed(4)}</strong><strong>{row.worst.toFixed(4)}</strong><strong>{row.threshold.toFixed(2)}</strong></div>)}</div>
  </section>;
}

function FinalComparison({ models }) {
  return <section className="panel audit-panel audit-comparison">
    <div className="audit-panel-head"><div><span className="eyebrow">FINAL COMPARISON</span><h3>All decision metrics</h3></div><span className="small-tag">Higher is better except Brier & errors</span></div>
    <div className="comparison-scroll"><table><thead><tr><th>Model</th><th>Threshold</th><th>Validation F1</th><th>Held-out F1</th><th>Worst dataset</th><th>ROC-AUC</th><th>Brier ↓</th><th>2026 F1</th><th>Errors</th></tr></thead><tbody>{models.map(model => <tr className={model.selected ? 'selected' : ''} key={model.id}><th><i style={{ background: model.colour }} />{model.shortName}{model.selected && <span>Preferred</span>}</th><td>{model.threshold.toFixed(2)}</td><td>{percent(model.validationMeanF1)}</td><td><b>{percent(model.testMeanF1)}</b></td><td>{percent(model.testWorstF1)}</td><td>{model.meanRocAuc.toFixed(4)}</td><td>{model.meanBrier.toFixed(4)}</td><td>{percent(model.contemporaryF1)}</td><td>{(model.errors.falsePositive + model.errors.falseNegative).toLocaleString()}</td></tr>)}</tbody></table></div>
  </section>;
}

export default function ModelAudit() {
  const [filter, setFilter] = useState('all');
  const visibleModels = useMemo(() => filter === 'all' ? trainingAudit.models : trainingAudit.models.filter(model => model.id === filter), [filter]);
  const selected = trainingAudit.models.find(model => model.selected);
  return <main className="page model-audit-page">
    <section className="audit-hero panel">
      <div><span className="eyebrow">AUDITED MODEL SNAPSHOT · {trainingAudit.auditDate.toUpperCase()}</span><h2>Three models. One honest comparison.</h2><p>Explore validation, held-out tests, calibration, contemporary claims, and model errors without hiding source-transfer weaknesses.</p><div className="audit-badges"><span>{trainingAudit.records.development.toLocaleString()} development records</span><span>{trainingAudit.records.test.toLocaleString()} held-out records</span><span>4 source datasets</span></div></div>
      <aside><span>Preferred baseline</span><strong>{selected.shortName}</strong><b>{percent(selected.testMeanF1)} mean test F1</b><small>Selected by source-held validation—not by the highest in-domain score.</small></aside>
    </section>

    <div className="audit-toolbar"><div><strong>Compare models</strong><span>All metrics are a fixed, reproducible audit snapshot.</span></div><div className="model-filter"><button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>All models</button>{trainingAudit.models.map(model => <button className={filter === model.id ? 'active' : ''} onClick={() => setFilter(model.id)} key={model.id}><i style={{ background: model.colour }} />{model.shortName}</button>)}</div></div>

    <section className="audit-kpis">{visibleModels.map(model => <article className={`panel audit-model-card ${model.selected ? 'selected' : ''}`} key={model.id} style={{ '--model-colour': model.colour }}><div><i /><span>{model.type}</span>{model.selected && <b>Preferred</b>}</div><h3>{model.shortName}</h3><strong>{percent(model.testMeanF1)}</strong><small>Mean held-out macro-F1</small><footer><span>Validation {percent(model.validationMeanF1)}</span><span>Contemporary {percent(model.contemporaryF1)}</span></footer></article>)}</section>

    <div className="audit-visual-grid"><PerformanceGraph models={visibleModels} /><DatasetGraph models={visibleModels} /></div>

    <div className="audit-two-column"><ValidationChart models={visibleModels} /><DatasetHeatmap models={visibleModels} /></div>
    <div className="audit-two-column"><Thresholds models={visibleModels} /><CalibrationPanel models={visibleModels} /></div>

    <section className="panel audit-panel audit-contemporary">
      <div className="audit-panel-head"><div><span className="eyebrow">UNSEEN 2026 DIAGNOSTIC</span><h3>20 contemporary claims</h3></div><span className="small-tag">10 real · 10 fake</span></div>
      <p className="audit-caption">A small diagnostic, never used for training or threshold selection. It is useful for direction—not a representative benchmark.</p>
      <div className="confusion-grid">{visibleModels.map(model => <ConfusionMatrix model={model} key={model.id} />)}</div>
    </section>

    <div className="audit-two-column audit-bottom"><ErrorTradeoff models={visibleModels} /><ForestTuning /></div>

    <FinalComparison models={visibleModels} />

    <section className="audit-method panel"><div><span className="eyebrow">READ THIS CORRECTLY</span><h3>Classification is not fact verification</h3></div><p>These models learn text patterns associated with dataset labels. High scores do not prove a claim is true or false. The future hybrid system must combine this signal with retrieved evidence, NLI verification, source context, and an explicit insufficient-evidence state.</p><a href="https://github.com/hardiktyagi032/hybrid-fake-news-detection/blob/main/docs/ml/classical_three_model_audit.md" target="_blank" rel="noreferrer">Read the complete methodology and limitations ↗</a></section>
    <ModelLegend models={trainingAudit.models} />
  </main>;
}
