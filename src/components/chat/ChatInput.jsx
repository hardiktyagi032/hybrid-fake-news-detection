import { useRef, useState } from 'react';
import Button from '../common/Button';

export default function ChatInput({ value, onChange, onSubmit, analysing }) {
  const inputRef = useRef();
  const fileRef = useRef();
  const [fileName, setFileName] = useState('');
  const submit = (event) => { event.preventDefault(); if (value.trim()) onSubmit(); };
  return <form onSubmit={submit} className="chat-input"><textarea ref={inputRef} value={value} onChange={(event) => onChange(event.target.value)} placeholder="Paste an article, claim, or URL to verify..." rows="4" /><input ref={fileRef} className="visually-hidden" type="file" accept=".pdf,.doc,.docx,.txt" onChange={(event) => setFileName(event.target.files?.[0]?.name || '')} /><div className="input-toolbar"><div><button type="button" className="input-tool" onClick={() => { onChange('https://'); inputRef.current?.focus(); }}>⌕ URL</button><button type="button" className="input-tool" onClick={() => fileRef.current?.click()}>⌑ Upload</button><button type="button" className="input-tool" onClick={() => { onChange(''); setFileName(''); }}>Clear</button><span className="input-hint">{fileName ? `Selected: ${fileName}` : 'We only retain this analysis in your history.'}</span></div><Button type="submit" disabled={analysing}>{analysing ? 'Analyzing...' : 'Analyze claim  →'}</Button></div></form>;
}
