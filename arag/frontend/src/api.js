import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const registerUser = (username, password) => {
  return fetch(`${API_URL}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
};

export const loginUser = (username, password) => {
  return fetch(`${API_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  });
};

export const getProfile = () => {
  const token = localStorage.getItem('token');
  return fetch(`${API_URL}/profile`, {
    headers: { 
      'Authorization': `Bearer ${token}`
    }
  });
};

export const updateProfile = (dietary_preferences) => {
  const token = localStorage.getItem('token');
  return fetch(`${API_URL}/profile`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ dietary_preferences })
  });
};

export const askQuestion = (token, query) => {
  return axios.post(`${API_URL}/ask`, { query }, {
    headers: { Authorization: `Bearer ${token}` }
  });
};

export const askQuestionStream = (query) => {
  const token = localStorage.getItem('token');
  const encodedQuery = encodeURIComponent(query);
  return `http://localhost:8000/rag/ask-stream?query=${encodedQuery}&token=${token}`;
};
