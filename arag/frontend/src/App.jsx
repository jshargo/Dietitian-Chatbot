import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom';
import Register from './components/Register.jsx';
import Login from './components/Login.jsx';
import Profile from './components/Profile.jsx';
import ProtectedRoute from './components/ProtectedRoute.jsx';
import './App.css';
function App() {
  const [username, setUsername] = useState(localStorage.getItem('username'));

  const handleLogin = (username) => {
    setUsername(username);
  };

  const handleLogout = () => {
    localStorage.removeItem('username');
    setUsername(null);
  };

  return (
    <Router>
      <div>
        <h1 className="welcome-header">Welcome!</h1>
        <nav className="auth-nav">
          {!username ? (
            <>
              <Link to="/register" className="auth-button">Register</Link>
              <Link to="/login" className="auth-button">Login</Link>
            </>
          ) : (
            <>
              <Link to="/profile" className="auth-button">Profile</Link>
              <button className="auth-button" onClick={handleLogout}>Logout</button>
            </>
          )}
        </nav>

        <Routes>
          <Route path="/register" element={<Register />} />
          <Route path="/login" element={<Login onLogin={handleLogin} />} />
          <Route 
            path="/profile" 
            element={
              <ProtectedRoute>
                <Profile username={username} />
              </ProtectedRoute>
            } 
          />
          <Route path="/" element={null} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
