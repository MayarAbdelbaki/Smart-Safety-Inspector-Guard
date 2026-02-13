/**
 * SSIG Admin Dashboard — Entry point.
 * Renders the React app (routing, auth, and Supabase-backed views are in App.js).
 */
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

