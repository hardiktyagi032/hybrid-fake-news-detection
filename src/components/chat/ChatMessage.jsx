export default function ChatMessage({ role = 'assistant', children }) {
  return <div className={`chat-message chat-message--${role}`}><div className="message-avatar">{role === 'assistant' ? 'V' : 'AC'}</div><div className="message-body">{children}</div></div>;
}
