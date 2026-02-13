/**
 * Guard View — list of unauthorized (unknown) persons from detections_log.
 * Filters rows where name === 'unknown person' and subscribes to real-time inserts.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import '../styles/Guard.css';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

function Guard() {
  const navigate = useNavigate();
  const [unknownPersons, setUnknownPersons] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check authentication via Supabase session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/login');
        return;
      }
    });

    // Load unknown persons
    loadUnknownPersons();

    // Subscribe to real-time changes for unknown persons
    const channel = supabase
      .channel('unknown_persons_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'detections_log',
          filter: 'name=eq.unknown person'
        },
        (payload) => {
          console.log('New unknown person detected:', payload);
          const newUnknown = payload.new;
          setUnknownPersons((prev) => [newUnknown, ...prev]);
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [navigate]);

  const loadUnknownPersons = async () => {
    try {
      const { data, error } = await supabase
        .from('detections_log')
        .select('*')
        .eq('name', 'unknown person')
        .order('created_at', { ascending: false })
        .limit(100);

      if (error) throw error;
      setUnknownPersons(data || []);
      setLoading(false);
    } catch (error) {
      console.error('Error loading unknown persons:', error);
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString();
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

  if (loading) {
    return (
      <div className="guard-container">
        <div className="loading">Loading unknown persons...</div>
      </div>
    );
  }

  return (
    <div className="guard-container">
      <header className="guard-header">
        <div className="header-content">
          <h1>Guard View - Unauthorized Persons</h1>
          <div className="header-actions">
            <button className="nav-button" onClick={() => navigate('/dashboard')}>
              Dashboard
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

      <div className="guard-info">
        <p className="info-text">
          This page displays all unauthorized persons detected by the system who are not registered in the member folder.
        </p>
        <div className="count-badge">
          Total Unknown Persons: <strong>{unknownPersons.length}</strong>
        </div>
      </div>

      <div className="unknown-persons-container">
        {unknownPersons.length === 0 ? (
          <div className="no-unknown-persons">
            <p>No unauthorized persons detected. All clear!</p>
          </div>
        ) : (
          <div className="unknown-persons-list">
            {/* Header Row */}
            <div className="unknown-person-row">
              <div className="unknown-cell">Image</div>
              <div className="unknown-cell">Status</div>
              <div className="unknown-cell">Detection Time</div>
            </div>
            
            {/* Data Rows */}
            {unknownPersons.map((person) => (
              <div key={person.id} className="unknown-person-row">
                <div className="person-image-cell">
                  {person.image_url ? (
                    <img 
                      src={person.image_url} 
                      alt="Unknown Person"
                    />
                  ) : (
                    <div className="no-image">No Image</div>
                  )}
                </div>
                
                <div className="unknown-cell unknown-name" data-label="Status">
                  <span className="alert-badge-inline">UNAUTHORIZED</span>
                </div>
                
                <div className="unknown-cell unknown-timestamp" data-label="Detection Time">
                  {formatTimestamp(person.created_at)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Guard;
