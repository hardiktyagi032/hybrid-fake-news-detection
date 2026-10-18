import { useState } from 'react';

export default function FollowUpInput() {
  const [text, setText] = useState('');
  const [messages, setMessages] = useState([]);
  const ask = (event) => { event.preventDefault(); if (!text.trim()) return; setMessages([...messages, text]); setText(''); };
  return <section className="follow-up panel"><div><span className="eyebrow">KEEP INVESTIGATING</span><h3>Ask about this analysis</h3></div>{messages.map(message => <div className="question-bubble" key={message}>{message}</div>)}<form onSubmit={ask}><input value={text} onChange={event => setText(event.target.value)} placeholder="Ask how a source supports the conclusion..." /><button aria-label="Send follow-up">↑</button></form></section>;
}
