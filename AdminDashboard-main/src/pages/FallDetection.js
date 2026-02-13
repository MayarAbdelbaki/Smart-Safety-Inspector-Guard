/**
 * Fall Detection — alerts from Supabase table `fall_alerts`.
 * Data is written by the Hugging Face app when the Roboflow fall model triggers.
 * Shows total/recent counts and a list with image_url or image_base64 and confidence.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import '../styles/FallDetection.css';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

function FallDetection() {
  const navigate = useNavigate();
  const [fallAlerts, setFallAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    total: 0,
    falling: 0,
    recent: 0
  });

  useEffect(() => {
    // Check authentication via Supabase session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/login');
        return;
      }
    });

    // Load fall alerts
    loadFallAlerts();

    // Subscribe to real-time changes
    const channel = supabase
      .channel('fall_alerts_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'fall_alerts'
        },
        (payload) => {
          console.log('New fall alert received:', payload);
          const newAlert = payload.new;
          setFallAlerts((prev) => [newAlert, ...prev]);
          updateStats(newAlert);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [navigate]);

  const loadFallAlerts = async () => {
    try {
      const { data, error } = await supabase
        .from('fall_alerts')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(100);

      if (error) throw error;
      setFallAlerts(data || []);
      
      // Calculate stats
      const total = data?.length || 0;
      const falling = data?.filter(alert => alert.is_falling === true).length || 0;
      setStats({ total, falling, recent: falling });
      
      setLoading(false);
    } catch (error) {
      console.error('Error loading fall alerts:', error);
      setLoading(false);
    }
  };

  const updateStats = (newAlert) => {
    setStats((prev) => ({
      total: prev.total + 1,
      falling: newAlert.is_falling ? prev.falling + 1 : prev.falling,
      recent: prev.recent + 1
    }));
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    const options = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    };
    const formatted = date.toLocaleString('en-US', options);
    return formatted.replace(',', ' at');
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      localStorage.removeItem('isAuthenticated');
      localStorage.removeItem('userEmail');
      localStorage.removeItem('userId');
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
      localStorage.removeItem('isAuthenticated');
      localStorage.removeItem('userEmail');
      localStorage.removeItem('userId');
      navigate('/login');
    }
  };

  if (loading) {
    return (
      <div className="fall-detection-container">
        <div className="loading">Loading fall alerts...</div>
      </div>
    );
  }

  return (
    <div className="fall-detection-container">
      <header className="fall-header">
        <div className="header-content">
          <h1>Fall Detection Alerts</h1>
          <div className="header-actions">
            <button className="nav-button" onClick={() => navigate('/dashboard')}>
              Dashboard
            </button>
            <button className="nav-button" onClick={() => navigate('/guard')}>
              Guard View
            </button>
            <button className="nav-button" onClick={() => navigate('/financial-penalty')}>
              Financial Penalty
            </button>
            <button className="logout-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="stats-container">
        <div className="stat-card">
          <h3>Total Alerts</h3>
          <p className="stat-number">{stats.total}</p>
        </div>
        <div className="stat-card">
          <h3>Falling Detected</h3>
          <p className="stat-number">{stats.falling}</p>
        </div>
        <div className="stat-card">
          <h3>Recent Alerts</h3>
          <p className="stat-number">{stats.recent}</p>
        </div>
      </div>

      <div className="fall-alerts-container">
        <div className="fall-alerts-header">
          <div>
            <h2>Recent Fall Alerts</h2>
            <p className="fall-alerts-subtitle">Live fall detection monitoring</p>
          </div>
          <button className="refresh-button" onClick={loadFallAlerts}>
            Refresh
          </button>
        </div>
        <div className="fall-alerts-list">
          {/* Header Row */}
          <div className="fall-alert-row header-row">
            <div className="fall-cell">TIME</div>
            <div className="fall-cell">STATUS</div>
            <div className="fall-cell">CONFIDENCE</div>
            <div className="fall-cell">IMAGE</div>
            <div className="fall-cell">DETAILS</div>
          </div>
          
          {/* Data Rows */}
          {fallAlerts.map((alert) => (
            <div key={alert.id} className="fall-alert-row">
              <div className="fall-cell fall-timestamp" data-label="TIME">
                {formatTimestamp(alert.created_at || alert.timestamp)}
              </div>
              
              <div className="fall-cell" data-label="STATUS">
                <span className={`status-badge ${alert.is_falling ? 'falling' : 'safe'}`}>
                  {alert.is_falling ? 'FALLING' : 'SAFE'}
                </span>
              </div>
              
              <div className="fall-cell" data-label="CONFIDENCE">
                {alert.confidence ? (
                  <span className="confidence-badge">
                    {(alert.confidence * 100).toFixed(1)}%
                  </span>
                ) : (
                  'N/A'
                )}
              </div>
              
              <div className="fall-cell fall-image-cell" data-label="IMAGE">
                {alert.image_url ? (
                  <img src={alert.image_url} alt="Fall Detection" className="fall-image" />
                ) : alert.image_base64 ? (
                  <img 
                    src={`data:image/jpeg;base64,${alert.image_base64}`} 
                    alt="Fall Detection" 
                    className="fall-image"
                  />
                ) : (
                  <div className="no-image">No Image</div>
                )}
              </div>
              
              <div className="fall-cell fall-details" data-label="DETAILS">
                {alert.raw_fall_detections && typeof alert.raw_fall_detections === 'object' ? (
                  <details className="raw-detections-details">
                    <summary>View Details</summary>
                    <pre>{JSON.stringify(alert.raw_fall_detections, null, 2)}</pre>
                  </details>
                ) : (
                  <span className="no-details">No additional details</span>
                )}
              </div>
            </div>
          ))}
        </div>
        
        {fallAlerts.length === 0 && (
          <div className="no-alerts">
            <p>No fall alerts detected yet.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default FallDetection;
