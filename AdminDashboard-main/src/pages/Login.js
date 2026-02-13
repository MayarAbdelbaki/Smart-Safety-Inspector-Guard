/**
 * Login page — Supabase Auth sign-in.
 * Uses REACT_APP_SUPABASE_* env vars only. No credentials stored in code.
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import '../styles/Login.css';
import logoRemoveBg from '../images/logo_removebg.png';

// Supabase: use environment variables only (no secrets in code).
// Set REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY in .env or your host's env.
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

const supabase = supabaseUrl && supabaseAnonKey
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Require Supabase env vars (set in .env or your deployment environment)
    if (!supabase || !supabaseUrl || !supabaseAnonKey) {
      setError('Supabase configuration missing. Set REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY in .env or your host environment.');
      setLoading(false);
      return;
    }

    try {
      // Sign in with Supabase Auth
      const { data, error: signInError } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password: password,
      });

      if (signInError) {
        console.error('Supabase Auth Error:', signInError);
        // Provide more helpful error messages
        if (signInError.message.includes('Invalid login credentials')) {
          setError('Invalid email or password. Please check your credentials.');
        } else if (signInError.message.includes('Email not confirmed')) {
          setError('Please confirm your email address first.');
        } else {
          setError(signInError.message || 'Invalid email or password');
        }
        setLoading(false);
        return;
      }

      if (data.user) {
        // Check if user is admin (optional - if you created admin_users table)
        // For now, any authenticated user can access
        // You can add admin check here if needed
        
        // Store session info
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('userEmail', data.user.email || email);
        localStorage.setItem('userId', data.user.id);
        
        navigate('/dashboard');
      }
    } catch (err) {
      console.error('Login error:', err);
      setError('An error occurred. Please check the browser console for details.');
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <img src={logoRemoveBg} alt="Logo" className="login-logo" />
          <h2>Admin Login</h2>
          <p>Please enter your email and password to access the system</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          {error && <div className="error-message">{error}</div>}
          
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter admin email"
              required
              autoFocus
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              required
            />
          </div>
          
          <button type="submit" className="login-button" disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default Login;
