export default function ErrorMessage({ children = 'Something went wrong. Please try again.' }) { return <p className="error-message" role="alert">{children}</p>; }
