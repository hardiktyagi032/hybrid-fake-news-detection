import ChatMessage from './ChatMessage';

export default function ChatWindow() {
  return <section className="chat-window"><ChatMessage><p>Hi, I’m Veritas. Share a news article, a link, or a claim and I’ll separate model signals from source-backed evidence.</p></ChatMessage><div className="suggestion-row"><span>Try an example</span><button>“The city approved a new water policy yesterday.”</button><button>Paste a news URL</button></div></section>;
}
