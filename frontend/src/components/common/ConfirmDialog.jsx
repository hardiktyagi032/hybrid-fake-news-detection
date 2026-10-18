import Modal from './Modal';
import Button from './Button';
export default function ConfirmDialog({ open, title = 'Are you sure?', onCancel, onConfirm }) { return <Modal open={open}><h3>{title}</h3><div className="dialog-actions"><Button variant="quiet" onClick={onCancel}>Cancel</Button><Button onClick={onConfirm}>Confirm</Button></div></Modal>; }
