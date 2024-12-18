import React, { useState, useEffect } from 'react';
import { askQuestion } from '../api';

function AskStream({ token }) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState('');

  const handleAsk = (e) => {
    e.preventDefault();
    setAnswer('');
    setError('');
    setIsAsking(true);

    const encodedQuery = encodeURIComponent(query);
    const eventSource = new EventSource(`http://localhost:8000/rag/ask-stream?query=${encodedQuery}`);

    eventSource.onmessage = (event) => {
      setAnswer((prev) => prev + event.data);
    };

    eventSource.onerror = (err) => {
      console.error("EventSource failed:", err);
      setError("An error occurred while fetching the answer.");
      eventSource.close();
      setIsAsking(false);
    };

    eventSource.onopen = () => {
      setIsAsking(false);
    };
  };

  return (
    <div>
      <h2>Ask a question</h2>
      <form onSubmit={handleAsk}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Your question"
          required
        />
        <button type="submit" disabled={isAsking}>Ask</button>
      </form>
      {error && <p className="error-message">{error}</p>}
      {answer && <p><strong>Answer:</strong> {answer}</p>}
      {isAsking && <p>Loading...</p>}
    </div>
  );
}

export default AskStream;