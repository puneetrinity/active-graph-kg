import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './lib/stores/authStore'
import Layout from './components/layout/Layout'
import Login from './routes/Login'
import Dashboard from './routes/Dashboard'
import Search from './routes/Search'
import Nodes from './routes/Nodes'

// Protected route wrapper
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }

  return <>{children}</>
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />

        {/* Protected routes with layout */}
        <Route
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/search" element={<Search />} />
          <Route path="/nodes" element={<Nodes />} />
        </Route>

        <Route path="/" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Router>
  )
}

export default App
