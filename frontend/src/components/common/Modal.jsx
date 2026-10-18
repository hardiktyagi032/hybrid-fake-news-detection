export default function Modal({ open, children }) { return open ? <div className="modal-backdrop"><div className="modal">{children}</div></div> : null; }
