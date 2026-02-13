/**
 * SSIG Admin Dashboard — Main app and routing.
 *
 * Uses Supabase for auth and data. Configure via environment variables:
 *   REACT_APP_SUPABASE_URL, REACT_APP_SUPABASE_ANON_KEY (see .env.example).
 *
 * Routes: / (welcome), /login, /dashboard (PPE), /guard, /fall-detection, /financial-penalty.
 * Protected routes require Supabase Auth session.
 */
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import Welcome from './pages/Welcome';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Guard from './pages/Guard';
import FallDetection from './pages/FallDetection';
import FinancialPenalty from './pages/FinancialPenalty';
import './App.css';

// Supabase client (env vars only; no secrets in code)
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

/** Wraps routes that require an authenticated Supabase session; redirects to /login otherwise. */
const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check current session
    supabase.auth.getSession().then(({ data: { session } }) => {
      setIsAuthenticated(!!session);
      setLoading(false);
    });

    // Listen for auth changes
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsAuthenticated(!!session);
      if (session) {
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('userEmail', session.user?.email || '');
        localStorage.setItem('userId', session.user?.id || '');
      } else {
        localStorage.removeItem('isAuthenticated');
        localStorage.removeItem('userEmail');
        localStorage.removeItem('userId');
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) {
    return <div style={{ padding: '20px', textAlign: 'center' }}>Loading...</div>;
  }

  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Welcome />} />
          <Route path="/login" element={<Login />} />
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/guard" 
            element={
              <ProtectedRoute>
                <Guard />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/fall-detection" 
            element={
              <ProtectedRoute>
                <FallDetection />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="/financial-penalty" 
            element={
              <ProtectedRoute>
                <FinancialPenalty />
              </ProtectedRoute>
            } 
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
