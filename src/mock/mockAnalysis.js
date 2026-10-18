export const stages = [
  ['Input processing', 'Article normalized and split into 14 sentences', 'done'],
  ['ML classification', 'Transformer and baseline models completed', 'done'],
  ['Evidence retrieval', '12 relevant reports found across 3 collections', 'done'],
  ['Fact verification', '4 claims checked against retrieved evidence', 'done'],
  ['Hybrid decision', 'Weighted fusion produced the final result', 'done'],
];

export const sources = [
  { id: 1, verdict: 'Supports', source: 'World Health Organization', type: 'Official health guidance', date: 'May 2024', relevance: 96, url: 'https://www.who.int/', excerpt: 'No evidence shows that routine vaccination changes human DNA. Vaccines train immune responses without entering the cell nucleus.' },
  { id: 2, verdict: 'Contradicts', source: 'Reuters Fact Check', type: 'Independent verification', date: 'Jun 2024', relevance: 93, url: 'https://www.reuters.com/fact-check/', excerpt: 'The viral post misrepresents how mRNA vaccines work and provides no evidence for its central claim.' },
  { id: 3, verdict: 'Supports', source: 'Nature Medicine', type: 'Peer-reviewed research', date: 'Apr 2024', relevance: 89, url: 'https://www.nature.com/nm/', excerpt: 'Researchers found no pathway by which vaccine mRNA integrates into the genome in normal clinical use.' },
];

export const history = [
  { title: 'Claim about vaccine ingredients', time: 'Today, 10:42 AM', verdict: 'Misleading', confidence: '92%' },
  { title: 'Election results report', time: 'Yesterday, 4:16 PM', verdict: 'Verified', confidence: '88%' },
  { title: 'Flood relief funding update', time: 'Oct 14, 2026', verdict: 'Insufficient evidence', confidence: '61%' },
  { title: 'New solar subsidy announcement', time: 'Oct 11, 2026', verdict: 'Verified', confidence: '90%' },
];
