/**
 * Financial Penalty — monthly penalty summary and per-person penalty history.
 * Reads from Supabase tables: financial_penalties, person_info (optional).
 * Subscribes to real-time inserts on financial_penalties. Configure tables in your Supabase project.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createClient } from '@supabase/supabase-js';
import '../styles/FinancialPenalty.css';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

function FinancialPenalty() {
  const navigate = useNavigate();
  const [selectedPersonId, setSelectedPersonId] = useState(null);
  const [penaltySummary, setPenaltySummary] = useState([]);
  const [personPenalties, setPersonPenalties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [personInfo, setPersonInfo] = useState(null);

  useEffect(() => {
    // Check authentication via Supabase session
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        navigate('/login');
        return;
      }
    });

    // Load penalty summary
    loadPenaltySummary();

    // Subscribe to real-time changes
    const channel = supabase
      .channel('penalties_changes')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'financial_penalties'
        },
        (payload) => {
          console.log('New penalty received:', payload);
          if (selectedPersonId) {
            loadPersonPenalties(selectedPersonId);
          }
          loadPenaltySummary();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [navigate, selectedPersonId]);

  const loadPenaltySummary = async () => {
    try {
      // Get current month's penalties grouped by person
      const startOfMonth = new Date();
      startOfMonth.setDate(1);
      startOfMonth.setHours(0, 0, 0, 0);

      const { data: penaltiesRaw, error: penaltiesError } = await supabase
        .from('financial_penalties')
        .select('*')
        .gte('violation_time', startOfMonth.toISOString())
        .order('violation_time', { ascending: false });

      if (penaltiesError) throw penaltiesError;

      const isValidPersonId = (pid) => {
        if (pid === null || pid === undefined) return false;
        const s = String(pid).trim();
        if (!s) return false;
        const lower = s.toLowerCase();
        if (lower === 'n/a' || lower === 'na' || lower === 'null' || lower === 'undefined') return false;
        return true;
      };

      const penalties = (penaltiesRaw || []).filter((p) => isValidPersonId(p.person_id));

      // Get unique person_ids from penalties
      const personIds = [...new Set(penalties.map(p => String(p.person_id).trim()))];

      // Get person info for each person_id
      const summaryPromises = personIds.map(async (personId) => {
        const { data: personData } = await supabase
          .from('person_info')
          .select('*')
          .eq('person_id', personId)
          .single();

        const personPenalties = penalties.filter(p => p.person_id === personId);
        const totalPenalties = personPenalties.reduce((sum, p) => sum + parseFloat(p.penalty_amount || 0), 0);
        const monthlySalary = personData?.monthly_salary || 5000;

        return {
          person_id: personId,
          name: personData?.name || `Person ${personId}`,
          monthly_salary: monthlySalary,
          total_penalties: totalPenalties,
          final_salary: monthlySalary - totalPenalties,
          penalty_count: personPenalties.length
        };
      });

      const summary = await Promise.all(summaryPromises);
      setPenaltySummary(summary);
      setLoading(false);
    } catch (error) {
      console.error('Error loading penalty summary:', error);
      setLoading(false);
    }
  };

  const loadPersonPenalties = async (personId) => {
    try {
      // Get person info
      const { data: personData } = await supabase
        .from('person_info')
        .select('*')
        .eq('person_id', personId)
        .single();

      setPersonInfo(personData || { person_id: personId, monthly_salary: 5000 });

      // Get all penalties for this person
      const { data: penalties, error } = await supabase
        .from('financial_penalties')
        .select('*')
        .eq('person_id', personId)
        .order('violation_time', { ascending: false })
        .limit(100);

      if (error) throw error;
      setPersonPenalties(penalties || []);
    } catch (error) {
      console.error('Error loading person penalties:', error);
    }
  };

  const handlePersonClick = (personId) => {
    setSelectedPersonId(personId);
    loadPersonPenalties(personId);
  };

  const handleBack = () => {
    setSelectedPersonId(null);
    setPersonPenalties([]);
    setPersonInfo(null);
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

  const formatCurrency = (amount) => {
    return `${parseFloat(amount || 0).toFixed(2)} EGY`;
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
      <div className="financial-penalty-container">
        <div className="loading">Loading financial penalties...</div>
      </div>
    );
  }

  // Show person detail view
  if (selectedPersonId) {
    const totalPenalties = personPenalties.reduce((sum, p) => sum + parseFloat(p.penalty_amount || 0), 0);
    const monthlySalary = personInfo?.monthly_salary || 5000;
    const finalSalary = monthlySalary - totalPenalties;

    return (
      <div className="financial-penalty-container">
        <header className="penalty-header">
          <div className="header-content">
            <div>
              <button className="back-button" onClick={handleBack}>
                ← Back to Summary
              </button>
              <h1>Financial Penalties - {personInfo?.name || `Person ${selectedPersonId}`}</h1>
            </div>
            <div className="header-actions">
              <button className="nav-button" onClick={() => navigate('/dashboard')}>
                Dashboard
              </button>
              <button className="nav-button" onClick={() => navigate('/guard')}>
                Guard View
              </button>
              <button className="nav-button" onClick={() => navigate('/fall-detection')}>
                Fall Detection
              </button>
              <button className="logout-button" onClick={handleLogout}>
                Logout
              </button>
            </div>
          </div>
        </header>

        <div className="person-summary-card">
          <div className="summary-item">
            <h3>Person ID</h3>
            <p className="summary-value">{selectedPersonId}</p>
          </div>
          <div className="summary-item">
            <h3>Monthly Salary</h3>
            <p className="summary-value salary">{formatCurrency(monthlySalary)}</p>
          </div>
          <div className="summary-item">
            <h3>Total Penalties</h3>
            <p className="summary-value penalty">{formatCurrency(totalPenalties)}</p>
          </div>
          <div className="summary-item">
            <h3>Final Salary</h3>
            <p className="summary-value final">{formatCurrency(finalSalary)}</p>
          </div>
        </div>

        <div className="penalties-container">
          <h2>Penalty History</h2>
          <div className="penalties-list">
            <div className="penalty-row header-row">
              <div className="penalty-cell">TIME</div>
              <div className="penalty-cell">VIOLATION TYPE</div>
              <div className="penalty-cell">PENALTY AMOUNT</div>
            </div>

            {personPenalties.map((penalty) => (
              <div key={penalty.id} className="penalty-row">
                <div className="penalty-cell penalty-timestamp" data-label="TIME">
                  {formatTimestamp(penalty.violation_time)}
                </div>
                <div className="penalty-cell" data-label="VIOLATION TYPE">
                  <span className="violation-badge">
                    {penalty.violation_type ? penalty.violation_type.replace(/_/g, ' ').toUpperCase() : 'MISSING PPE'}
                  </span>
                </div>
                <div className="penalty-cell penalty-amount" data-label="PENALTY AMOUNT">
                  {formatCurrency(penalty.penalty_amount)}
                </div>
              </div>
            ))}

            {personPenalties.length === 0 && (
              <div className="no-penalties">
                <p>No penalties recorded for this person.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Show summary view
  return (
    <div className="financial-penalty-container">
      <header className="penalty-header">
        <div className="header-content">
          <h1>Financial Penalty System</h1>
          <div className="header-actions">
            <button className="nav-button" onClick={() => navigate('/dashboard')}>
              Dashboard
            </button>
            <button className="nav-button" onClick={() => navigate('/guard')}>
              Guard View
            </button>
            <button className="nav-button" onClick={() => navigate('/fall-detection')}>
              Fall Detection
            </button>
            <button className="logout-button" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="penalty-info">
        <p className="info-text">
          Financial penalties are automatically applied when a person is detected without 2 or more PPE items (Helmet, Vest, Mask).
          Each violation results in a 50 EGY penalty, with a 30-minute cooldown period between penalties.
        </p>
      </div>

      <div className="penalty-summary-container">
        <h2>Monthly Penalty Summary</h2>
        <div className="summary-list">
          <div className="summary-row header-row">
            <div className="summary-cell">PERSON ID</div>
            <div className="summary-cell">NAME</div>
            <div className="summary-cell">MONTHLY SALARY</div>
            <div className="summary-cell">TOTAL PENALTIES</div>
            <div className="summary-cell">FINAL SALARY</div>
            <div className="summary-cell">ACTIONS</div>
          </div>

          {penaltySummary.map((person) => (
            <div key={person.person_id} className="summary-row">
              <div className="summary-cell" data-label="PERSON ID">
                {person.person_id}
              </div>
              <div className="summary-cell" data-label="NAME">
                {person.name}
              </div>
              <div className="summary-cell" data-label="MONTHLY SALARY">
                {formatCurrency(person.monthly_salary)}
              </div>
              <div className="summary-cell penalty-amount" data-label="TOTAL PENALTIES">
                {formatCurrency(person.total_penalties)} ({person.penalty_count} violations)
              </div>
              <div className="summary-cell final-salary" data-label="FINAL SALARY">
                {formatCurrency(person.final_salary)}
              </div>
              <div className="summary-cell" data-label="ACTIONS">
                <button 
                  className="view-details-button" 
                  onClick={() => handlePersonClick(person.person_id)}
                >
                  View Details
                </button>
              </div>
            </div>
          ))}

          {penaltySummary.length === 0 && (
            <div className="no-summary">
              <p>No penalties recorded this month.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FinancialPenalty;
