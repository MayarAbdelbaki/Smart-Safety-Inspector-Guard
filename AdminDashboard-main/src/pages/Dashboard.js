/**
 * PPE Detection Dashboard — main view of detections from Supabase.
 *
 * Reads from `detections_log` (same table the Hugging Face app writes to).
 * Shows: total detections, helmet/vest/mask counts, unknown persons, and a
 * real-time list of recent detections (person, PPE status, compliance).
 * Uses Supabase Realtime to append new rows as they are inserted.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import '../styles/Dashboard.css';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

function Dashboard() {
  const navigate = useNavigate();
  const [detections, setDetections] = useState([]);
  const [stats, setStats] = useState({
    total: 0,
    withHelmet: 0,
    withVest: 0,
    withMask: 0,
    unknown: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Require auth; redirect to login if no session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/login');
        return;
      }
    });

    loadDetections();
    loadStats();

    // Supabase Realtime: append new detections when Hugging Face app inserts into detections_log
    const channel = supabase
      .channel('detections_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'detections_log'
        },
        (payload) => {
          console.log('New detection received:', payload);
          const newDetection = payload.new;
          setDetections((prev) => [newDetection, ...prev]);
          updateStats(newDetection);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [navigate]);

  const loadDetections = async () => {
    try {
      const { data, error } = await supabase
        .from('detections_log')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(50);

      if (error) throw error;
      setDetections(data || []);
      setLoading(false);
    } catch (error) {
      console.error('Error loading detections:', error);
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const { count } = await supabase
        .from('detections_log')
        .select('*', { count: 'exact', head: true });

      const recent = await supabase
        .from('detections_log')
        .select('helmet, vest, mask, name')
        .order('created_at', { ascending: false })
        .limit(100);

      if (recent.data) {
        setStats({
          total: count || 0,
          withHelmet: recent.data.filter(d => d.helmet).length,
          withVest: recent.data.filter(d => d.vest).length,
          withMask: recent.data.filter(d => d.mask).length,
          unknown: recent.data.filter(d => d.name === 'unknown person').length
        });
      }
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const updateStats = (newDetection) => {
    setStats((prev) => ({
      total: prev.total + 1,
      withHelmet: newDetection.helmet ? prev.withHelmet + 1 : prev.withHelmet,
      withVest: newDetection.vest ? prev.withVest + 1 : prev.withVest,
      withMask: newDetection.mask ? prev.withMask + 1 : prev.withMask,
      unknown: newDetection.name === 'unknown person' ? prev.unknown + 1 : prev.unknown
    }));
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
      // Still clear local storage and navigate
      localStorage.removeItem('isAuthenticated');
      localStorage.removeItem('userEmail');
      localStorage.removeItem('userId');
      navigate('/login');
    }
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

  /** Maps DB value to display state: true (wearing), false (not wearing), null (unknown/not detected). */
  const getPPEStatus = (value) => {
    // Strict null/undefined check — must be first
    if (value === null || value === undefined) {
      return 'null';
    }
    // Check if it's explicitly the string "null"
    if (String(value).toLowerCase() === 'null') {
      return 'null';
    }
    // Check if it's a boolean false (explicit false, not null)
    if (value === false) {
      return 'false';
    }
    // Check if it's a boolean true
    if (value === true) {
      return 'true';
    }
    // For any other value, return null
    return 'null';
  };

  /** Computes compliance label: Compliant (all PPE true), Violation (any false), else Unknown. */
  const getOverallStatus = (detection) => {
    const helmet = detection.helmet;
    const vest = detection.vest;
    const mask = detection.mask;
    if (helmet === false || vest === false || mask === false) return 'Violation';
    if (helmet === true && vest === true && mask === true) return 'Compliant';
    return 'Unknown';
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading detections...</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-content">
          <h1>PPE Detection Dashboard</h1>
          <div className="header-actions">
            <button className="nav-button" onClick={() => navigate('/guard')}>
              Guard View
            </button>
            <button className="nav-button" onClick={() => navigate('/fall-detection')}>
              Fall Detection
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
          <h3>Total Detections</h3>
          <p className="stat-number">{stats.total}</p>
        </div>
        <div className="stat-card">
          <h3>With Helmet</h3>
          <p className="stat-number">{stats.withHelmet}</p>
        </div>
        <div className="stat-card">
          <h3>With Vest</h3>
          <p className="stat-number">{stats.withVest}</p>
        </div>
        <div className="stat-card">
          <h3>With Mask</h3>
          <p className="stat-number">{stats.withMask}</p>
        </div>
        <div className="stat-card">
          <h3>Unknown Persons</h3>
          <p className="stat-number">{stats.unknown}</p>
        </div>
      </div>

      <div className="detections-container">
        <div className="detections-header">
          <div>
            <h2>Recent Detections</h2>
            <p className="detections-subtitle">Live updates from camera system</p>
          </div>
          <button className="refresh-button" onClick={loadDetections}>
            Refresh
          </button>
        </div>
        <div className="detections-list">
          {/* Header Row */}
          <div className="detection-row header-row">
            <div className="detection-cell">TIME</div>
            <div className="detection-cell">PERSON</div>
            <div className="detection-cell">ID</div>
            <div className="detection-cell">HELMET</div>
            <div className="detection-cell">VEST</div>
            <div className="detection-cell">MASK</div>
            <div className="detection-cell">STATUS</div>
            <div className="detection-cell">CONFIDENCE</div>
          </div>
          
          {/* Data Rows */}
          {detections.map((detection) => (
            <div key={detection.id} className="detection-row">
              <div className="detection-cell detection-timestamp" data-label="TIME">
                <span className="icon-clock"></span>
                {formatTimestamp(detection.created_at)}
              </div>
              
              <div className="detection-cell detection-name" data-label="PERSON">
                <span className="icon-person"></span>
                {detection.name || 'Unknown'}
              </div>
              
              <div className="detection-cell detection-person-id" data-label="ID">
                {detection.person_id || 'N/A'}
              </div>
              
              <div className="detection-cell" data-label="HELMET">
                <div className={`ppe-icon ${getPPEStatus(detection.helmet)}`}>
                  {getPPEStatus(detection.helmet) === 'true' && <span className="check-icon">✓</span>}
                  {getPPEStatus(detection.helmet) === 'false' && <span className="x-icon">✗</span>}
                  {getPPEStatus(detection.helmet) === 'null' && <span className="null-icon">○</span>}
                </div>
              </div>
              
              <div className="detection-cell" data-label="VEST">
                <div className={`ppe-icon ${getPPEStatus(detection.vest)}`}>
                  {getPPEStatus(detection.vest) === 'true' && <span className="check-icon">✓</span>}
                  {getPPEStatus(detection.vest) === 'false' && <span className="x-icon">✗</span>}
                  {getPPEStatus(detection.vest) === 'null' && <span className="null-icon">○</span>}
                </div>
              </div>
              
              <div className="detection-cell" data-label="MASK">
                <div className={`ppe-icon ${getPPEStatus(detection.mask)}`}>
                  {getPPEStatus(detection.mask) === 'true' && <span className="check-icon">✓</span>}
                  {getPPEStatus(detection.mask) === 'false' && <span className="x-icon">✗</span>}
                  {getPPEStatus(detection.mask) === 'null' && <span className="null-icon">○</span>}
                </div>
              </div>
              
              <div className="detection-cell" data-label="STATUS">
                <span className={`status-badge ${getOverallStatus(detection).toLowerCase()}`}>
                  {getOverallStatus(detection)}
                </span>
              </div>
              
              <div className="detection-cell" data-label="CONFIDENCE">
                {detection.confidence ? (detection.confidence * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
          ))}
        </div>
        
        {detections.length === 0 && (
          <div className="no-detections">
            <p>No detections yet. Waiting for camera input...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
