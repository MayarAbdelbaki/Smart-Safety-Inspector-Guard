/**
 * Welcome / landing page. Entry point before login; "Enter System" navigates to /login.
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Welcome.css';
import logoRemoveBg from '../images/logo_removebg.png';

function Welcome() {
  const navigate = useNavigate();

  const handleEnter = () => {
    navigate('/login');
  };

  return (
    <div className="welcome-container">
      <div className="welcome-content">
        <div className="logo-container">
          <img src={logoRemoveBg} alt="Company Logo" className="welcome-logo" />
        </div>
        <h1 className="welcome-title">PPE Detection System</h1>
        <p className="welcome-subtitle">Advanced Personal Protective Equipment Monitoring</p>
        <button className="enter-button" onClick={handleEnter}>
          Enter System
        </button>
      </div>
    </div>
  );
}

export default Welcome;
