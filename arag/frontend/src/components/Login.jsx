import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.token);
        localStorage.setItem('username', data.username);
        onLogin(data.username);
        navigate('/profile');
      } else {
        setError('Login failed');
      }
    } catch (err) {
      setError('Error connecting to server');
    }
  };

  return (
    <div className="auth-form">
      <h2>Login</h2>
      {error && <p className="error-message">{error}</p>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <input 
            type="text"
            value={username} 
            onChange={e => setUsername(e.target.value)} 
            placeholder="Username"
            required
            className="auth-input"
          />
        </div>
        <div className="form-group">
          <input 
            type="password"
            value={password} 
            onChange={e => setPassword(e.target.value)} 
            placeholder="Password"
            required
            className="auth-input"
          />
        </div>
        <button type="submit" className="auth-button">Login</button>
      </form>
    </div>
  );
}

export default Login;
