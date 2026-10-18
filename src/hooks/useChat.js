import { useState } from 'react';
export function useChat() { const [messages, setMessages] = useState([]); const addMessage = (message) => setMessages(current => [...current, message]); return { messages, addMessage }; }
