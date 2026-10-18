const findings = [
  ['01', 'The central claim conflicts with established biological evidence.'],
  ['02', 'Two independent fact-checks identify the claim as false or misleading.'],
  ['03', 'Model signals agree strongly with the evidence-verification result.'],
];
export default function KeyFindings() { return <section className="key-findings panel"><span className="eyebrow">KEY FINDINGS</span>{findings.map(([number, finding]) => <div className="finding" key={number}><span>{number}</span><p>{finding}</p></div>)}</section>; }
