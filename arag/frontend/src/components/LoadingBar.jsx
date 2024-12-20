import React from 'react';
import './LoadingBar.css';

function LoadingBar({ isLoading }) {
  if (!isLoading) return null;
  
  return (
    <div className="loading-container">
      <div className="loading-bar">
        <div className="loading-progress"></div>
      </div>
      <p className="loading-text">Generating response...</p>
    </div>
  );
}

export default LoadingBar; 