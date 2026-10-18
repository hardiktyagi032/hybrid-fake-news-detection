import { useState } from 'react';
export function useHistory(initial = []) { const [items, setItems] = useState(initial); return { items, setItems }; }
