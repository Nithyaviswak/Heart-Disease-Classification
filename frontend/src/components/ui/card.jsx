export function Card({ children, className = "" }) {
  return <div className={`bg-white rounded-xl border border-slate-200 ${className}`}>{children}</div>;
}

export function CardContent({ children, className = "" }) {
  return <div className={`p-5 ${className}`}>{children}</div>;
}
