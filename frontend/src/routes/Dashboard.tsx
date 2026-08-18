import { useAuthStore } from '../lib/stores/authStore'

export default function Dashboard() {
  const { claims } = useAuthStore()

  return (
    <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">
          Welcome to Active Graph KG
        </p>
      </div>

      {/* User Info Card */}
      {claims && (
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">User Information</h2>
          <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <dt className="text-sm font-medium text-gray-500">Name</dt>
              <dd className="text-sm text-gray-900 mt-1">{claims.name || 'N/A'}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Email</dt>
              <dd className="text-sm text-gray-900 mt-1">{claims.email || 'N/A'}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Tenant ID</dt>
              <dd className="text-sm text-gray-900 mt-1">{claims.tenant_id}</dd>
            </div>
            <div>
              <dt className="text-sm font-medium text-gray-500">Scopes</dt>
              <dd className="text-sm text-gray-900 mt-1">
                {claims.scopes.join(', ')}
              </dd>
            </div>
          </dl>
        </div>
      )}

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <a
          href="/search"
          className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Search</h3>
          <p className="text-sm text-gray-600">
            Hybrid search across the knowledge graph using vector similarity and text matching
          </p>
        </a>

        <a
          href="/nodes"
          className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Nodes</h3>
          <p className="text-sm text-gray-600">
            Create, view, edit, and manage nodes in your knowledge graph
          </p>
        </a>
      </div>
    </div>
  )
}
