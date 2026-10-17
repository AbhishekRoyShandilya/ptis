// Generic loading skeleton for pages and cards
interface SkeletonProps {
  lines?: number;
  className?: string;
}

export function CardSkeleton({ lines = 3, className = "" }: SkeletonProps) {
  return (
    <div className={`animate-pulse bg-gray-800 rounded-lg p-4 ${className}`}>
      <div className="h-4 bg-gray-700 rounded w-1/3 mb-3" />
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className="h-3 bg-gray-700 rounded mb-2"
          style={{ width: `${70 + Math.random() * 30}%` }}
        />
      ))}
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="animate-pulse space-y-2">
      <div className="h-8 bg-gray-700 rounded" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-10 bg-gray-800 rounded" />
      ))}
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="p-6 space-y-4">
      <div className="animate-pulse h-8 bg-gray-700 rounded w-1/4" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <CardSkeleton />
        <CardSkeleton />
        <CardSkeleton />
      </div>
      <TableSkeleton />
    </div>
  );
}
