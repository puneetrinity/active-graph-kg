import { Link, useLocation, Outlet } from 'react-router-dom'
import { useAuthStore } from '../../lib/stores/authStore'
import { useRateLimitStore } from '../../lib/stores/rateLimitStore'
import { cn } from '../../lib/utils/cn'

export default function Layout() {
  const location = useLocation()
  const { claims, logout } = useAuthStore()
  const { remaining, limit, isWarning } = useRateLimitStore()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'Search', href: '/search' },
    { name: 'Nodes', href: '/nodes' },
  ]

  const isActive = (path: string) => location.pathname === path

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo & Nav */}
            <div className="flex items-center gap-8">
              <h1 className="text-xl font-bold text-gray-900">Active Graph KG</h1>
              <nav className="flex gap-4">
                {navigation.map((item) => (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={cn(
                      'px-3 py-2 rounded-md text-sm font-medium transition-colors',
                      isActive(item.href)
                        ? 'bg-gray-900 text-white'
                        : 'text-gray-700 hover:bg-gray-100'
                    )}
                  >
                    {item.name}
                  </Link>
                ))}
              </nav>
            </div>

            {/* User Info & Rate Limit */}
            <div className="flex items-center gap-4">
              {/* Rate Limit Badge */}
              {remaining !== null && (
                <div
                  className={cn(
                    'px-3 py-1 rounded-full text-xs font-medium',
                    isWarning
                      ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                      : 'bg-gray-100 text-gray-700'
                  )}
                >
                  {remaining}/{limit} requests
                </div>
              )}

              {/* Tenant Badge */}
              {claims?.tenant_id && (
                <div className="px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  Tenant: {claims.tenant_id}
                </div>
              )}

              {/* Logout */}
              <button
                onClick={logout}
                className="text-sm text-gray-700 hover:text-gray-900"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main>
        <Outlet />
      </main>
    </div>
  )
}
