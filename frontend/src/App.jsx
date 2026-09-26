import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

// Import your two pages (make sure the file paths match where they are saved!)
import SatQueryFrontend from './SatQueryFrontend';
import Dashboard from './Dashboard';
import { AuthProvider } from './auth/AuthContext';
import RequireAuth from './auth/RequireAuth';
import AuthPage from './auth/AuthPage';


function App() {
  return (
    <Router>
      <AuthProvider>
      <Routes>
        {/* When the URL is exactly "/", load the 3D Globe home page */}
        <Route path="/" element={<SatQueryFrontend />} />
        
        <Route path="/login" element={<AuthPage key="login" mode="login" />} />
        <Route path="/signup" element={<AuthPage key="signup" mode="signup" />} />

        {/* The dashboard needs an account */}
        <Route path="/dashboard" element={<RequireAuth><Dashboard /></RequireAuth>} />
      </Routes>
      </AuthProvider>
    </Router>
  );
}

export default App;