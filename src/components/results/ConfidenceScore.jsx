export default function ConfidenceScore({ score = 92 }) {
  const dash = 251.2 - (251.2 * score) / 100;
  return <div className="confidence-score"><svg viewBox="0 0 100 100"><circle className="score-bg" cx="50" cy="50" r="40" /><circle className="score-value" cx="50" cy="50" r="40" style={{ strokeDashoffset: dash }} /></svg><div><strong>{score}%</strong><span>confidence</span></div></div>;
}
