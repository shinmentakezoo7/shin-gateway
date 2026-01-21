'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Server,
  Cpu,
  Key,
  BarChart3,
  ArrowLeftRight,
  Settings,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/', label: 'Overview', icon: LayoutDashboard, description: 'Dashboard home' },
  { href: '/providers', label: 'Providers', icon: Server, description: 'Manage backends' },
  { href: '/models', label: 'Models', icon: Cpu, description: 'Model aliases' },
  { href: '/api-keys', label: 'API Keys', icon: Key, description: 'Access tokens' },
  { href: '/statistics', label: 'Statistics', icon: BarChart3, description: 'Usage analytics' },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-72 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center">
              <ArrowLeftRight className="w-5 h-5 text-black" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-white border-2 border-card flex items-center justify-center">
              <div className="w-1.5 h-1.5 rounded-full bg-black" />
            </div>
          </div>
          <div>
            <h1 className="text-lg font-bold text-foreground">Shin Gateway</h1>
            <p className="text-xs text-muted-foreground">Protocol Translator</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider px-3 mb-3">
          Navigation
        </p>
        {navItems.map((item, index) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'group flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200',
                'animate-slide-in',
                isActive
                  ? 'bg-white/10 border border-white/20'
                  : 'hover:bg-accent border border-transparent'
              )}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div
                className={cn(
                  'w-9 h-9 rounded-lg flex items-center justify-center transition-all duration-200',
                  isActive
                    ? 'bg-white text-black'
                    : 'bg-secondary text-muted-foreground group-hover:bg-accent group-hover:text-foreground'
                )}
              >
                <item.icon className="w-4 h-4" />
              </div>
              <div className="flex-1 min-w-0">
                <p
                  className={cn(
                    'text-sm font-medium truncate',
                    isActive ? 'text-foreground' : 'text-muted-foreground group-hover:text-foreground'
                  )}
                >
                  {item.label}
                </p>
                <p className="text-xs text-muted-foreground truncate">{item.description}</p>
              </div>
              <ChevronRight
                className={cn(
                  'w-4 h-4 transition-all duration-200',
                  isActive
                    ? 'text-foreground opacity-100'
                    : 'text-muted-foreground opacity-0 group-hover:opacity-100'
                )}
              />
            </Link>
          );
        })}
      </nav>

      {/* Status Footer */}
      <div className="p-4 border-t border-border">
        <div className="rounded-xl border border-border bg-secondary p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-2 h-2 rounded-full status-online" />
            <span className="text-xs font-medium text-foreground">System Online</span>
          </div>
          <div className="space-y-2 text-xs text-muted-foreground">
            <div className="flex justify-between">
              <span>Version</span>
              <span className="text-foreground">1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span>API Status</span>
              <span className="text-foreground">Healthy</span>
            </div>
          </div>
        </div>

        {/* Settings link */}
        <Link
          href="/settings"
          className="flex items-center gap-2 px-3 py-2 mt-3 text-muted-foreground hover:text-foreground transition-colors rounded-lg hover:bg-accent"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">Settings</span>
        </Link>
      </div>
    </aside>
  );
}
