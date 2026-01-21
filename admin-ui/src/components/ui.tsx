'use client';

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { LucideIcon } from 'lucide-react';

// Button Component
const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 btn-shine',
  {
    variants: {
      variant: {
        default: 'bg-white text-black hover:bg-white/90',
        secondary: 'bg-secondary text-secondary-foreground border border-border hover:bg-accent',
        outline: 'border border-border bg-transparent hover:bg-accent hover:text-accent-foreground',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        destructive: 'bg-white/10 text-white hover:bg-white/20 border border-white/20',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-12 rounded-lg px-8',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  icon?: LucideIcon;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, icon: Icon, children, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button';
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props}>
        {Icon && <Icon className="h-4 w-4" />}
        {children}
      </Comp>
    );
  }
);
Button.displayName = 'Button';

// Card Component
const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement> & { noPadding?: boolean }>(
  ({ className, noPadding, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        'rounded-xl border border-border bg-card text-card-foreground',
        className
      )}
      {...props}
    />
  )
);
Card.displayName = 'Card';

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn('flex flex-col space-y-1.5 p-5 pb-0', className)}
      {...props}
    />
  )
);
CardHeader.displayName = 'CardHeader';

const CardTitle = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn('text-lg font-semibold leading-none tracking-tight', className)}
      {...props}
    />
  )
);
CardTitle.displayName = 'CardTitle';

const CardDescription = React.forwardRef<HTMLParagraphElement, React.HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
  )
);
CardDescription.displayName = 'CardDescription';

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn('p-5', className)} {...props} />
  )
);
CardContent.displayName = 'CardContent';

// Input Component
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  icon?: LucideIcon;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, label, icon: Icon, ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && <label className="text-sm font-medium text-muted-foreground">{label}</label>}
        <div className="relative">
          {Icon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
              <Icon className="h-4 w-4" />
            </div>
          )}
          <input
            type={type}
            className={cn(
              'flex h-10 w-full rounded-lg border border-input bg-secondary px-3 py-2 text-sm ring-offset-background',
              'file:border-0 file:bg-transparent file:text-sm file:font-medium',
              'placeholder:text-muted-foreground',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
              'disabled:cursor-not-allowed disabled:opacity-50',
              Icon && 'pl-10',
              className
            )}
            ref={ref}
            {...props}
          />
        </div>
      </div>
    );
  }
);
Input.displayName = 'Input';

// Select Component
interface SelectProps {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
}

const Select = ({ label, value, onChange, options }: SelectProps) => {
  return (
    <div className="space-y-2">
      {label && <label className="text-sm font-medium text-muted-foreground">{label}</label>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          'flex h-10 w-full rounded-lg border border-input bg-secondary px-3 py-2 text-sm ring-offset-background',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50'
        )}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value} className="bg-card">
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};

// Badge Component
const badgeVariants = cva(
  'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-white text-black',
        secondary: 'border-border bg-secondary text-secondary-foreground',
        outline: 'border-border text-foreground',
        muted: 'border-transparent bg-muted text-muted-foreground',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {
  dot?: boolean;
  online?: boolean;
}

const Badge = ({ className, variant, dot, online, children, ...props }: BadgeProps) => {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props}>
      {dot && (
        <span className={cn('h-1.5 w-1.5 rounded-full', online ? 'status-online' : 'bg-muted-foreground')} />
      )}
      {children}
    </div>
  );
};

// StatCard Component
interface StatCardProps {
  title: string;
  value: string | number;
  suffix?: string;
  icon?: LucideIcon;
  description?: string;
}

const StatCard = ({ title, value, suffix, icon: Icon, description }: StatCardProps) => {
  return (
    <div className="rounded-xl border border-border bg-card p-5 card-interactive">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-sm text-muted-foreground">{title}</p>
          <p className="text-3xl font-bold stat-number">
            {value}
            {suffix && <span className="text-lg text-muted-foreground ml-0.5">{suffix}</span>}
          </p>
          {description && <p className="text-xs text-muted-foreground">{description}</p>}
        </div>
        {Icon && (
          <div className="rounded-lg bg-white/5 p-2.5 border border-border">
            <Icon className="h-5 w-5 text-white" />
          </div>
        )}
      </div>
    </div>
  );
};

// EmptyState Component
interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

const EmptyState = ({ icon: Icon, title, description, action }: EmptyStateProps) => {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      {Icon && (
        <div className="rounded-xl bg-secondary p-4 mb-4 border border-border">
          <Icon className="h-8 w-8 text-muted-foreground" />
        </div>
      )}
      <h3 className="text-lg font-medium mb-1">{title}</h3>
      {description && <p className="text-sm text-muted-foreground mb-4 max-w-sm">{description}</p>}
      {action}
    </div>
  );
};

// LoadingSpinner Component
const LoadingSpinner = ({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) => {
  const sizes = { sm: 'h-4 w-4', md: 'h-8 w-8', lg: 'h-12 w-12' };
  return (
    <div className="flex items-center justify-center">
      <div className={cn('border-2 border-border border-t-white rounded-full animate-spin', sizes[size])} />
    </div>
  );
};

// PageHeader Component
interface PageHeaderProps {
  title: string;
  description?: string;
  action?: React.ReactNode;
}

const PageHeader = ({ title, description, action }: PageHeaderProps) => {
  return (
    <div className="flex items-start justify-between mb-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
        {description && <p className="text-muted-foreground mt-1">{description}</p>}
      </div>
      {action}
    </div>
  );
};

// Separator Component
const Separator = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn('h-px bg-border', className)} {...props} />
);

// Progress Component
interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
  max?: number;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showValue?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, max = 100, variant = 'default', showValue = false, size = 'md', ...props }, ref) => {
    const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
    const sizes = { sm: 'h-1.5', md: 'h-2.5', lg: 'h-4' };
    const variants = {
      default: 'bg-white',
      success: 'bg-white',
      warning: 'bg-white/70',
      danger: 'bg-white/50',
    };

    return (
      <div className={cn('relative w-full', className)} ref={ref} {...props}>
        <div className={cn('w-full rounded-full bg-secondary border border-border overflow-hidden', sizes[size])}>
          <div
            className={cn('h-full rounded-full transition-all duration-500 ease-out', variants[variant])}
            style={{ width: `${percentage}%` }}
          />
        </div>
        {showValue && (
          <span className="absolute right-0 -top-5 text-xs text-muted-foreground">
            {value.toLocaleString()} / {max.toLocaleString()}
          </span>
        )}
      </div>
    );
  }
);
Progress.displayName = 'Progress';

// MetricCard Component - Enhanced stat card for real-time metrics
interface MetricCardProps {
  title: string;
  value: string | number;
  suffix?: string;
  icon?: LucideIcon;
  trend?: { value: number; label: string };
  status?: 'healthy' | 'warning' | 'danger';
  subValue?: string;
  progress?: { value: number; max: number };
  animate?: boolean;
}

const MetricCard = ({
  title,
  value,
  suffix,
  icon: Icon,
  trend,
  status,
  subValue,
  progress,
  animate = true,
}: MetricCardProps) => {
  const statusColors = {
    healthy: 'border-white/30',
    warning: 'border-white/20',
    danger: 'border-white/10',
  };
  const statusDotColors = {
    healthy: 'status-online',
    warning: 'bg-white/50',
    danger: 'bg-white/30 animate-pulse',
  };

  return (
    <div
      className={cn(
        'rounded-xl border bg-card p-5 card-interactive relative overflow-hidden',
        status ? statusColors[status] : 'border-border',
        animate && 'animate-fade-in'
      )}
    >
      {status && (
        <div className="absolute top-3 right-3">
          <div className={cn('w-2 h-2 rounded-full', statusDotColors[status])} />
        </div>
      )}
      <div className="flex items-start justify-between">
        <div className="space-y-1 flex-1">
          <div className="flex items-center gap-2">
            <p className="text-sm text-muted-foreground">{title}</p>
            {trend && (
              <span
                className={cn(
                  'text-xs px-1.5 py-0.5 rounded-md',
                  trend.value >= 0 ? 'bg-white/10 text-white/80' : 'bg-white/5 text-white/50'
                )}
              >
                {trend.value >= 0 ? '+' : ''}
                {trend.value}% {trend.label}
              </span>
            )}
          </div>
          <p className="text-3xl font-bold stat-number">
            {value}
            {suffix && <span className="text-lg text-muted-foreground ml-0.5">{suffix}</span>}
          </p>
          {subValue && <p className="text-xs text-muted-foreground">{subValue}</p>}
          {progress && (
            <div className="mt-3">
              <Progress value={progress.value} max={progress.max} size="sm" />
              <p className="text-xs text-muted-foreground mt-1">
                {Math.round((progress.value / progress.max) * 100)}% of limit
              </p>
            </div>
          )}
        </div>
        {Icon && (
          <div className="rounded-lg bg-white/5 p-2.5 border border-border">
            <Icon className="h-5 w-5 text-white" />
          </div>
        )}
      </div>
    </div>
  );
};

export {
  Button,
  buttonVariants,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  Input,
  Select,
  Badge,
  badgeVariants,
  StatCard,
  EmptyState,
  LoadingSpinner,
  PageHeader,
  Separator,
  Progress,
  MetricCard,
};
