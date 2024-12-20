import React, { useState } from 'react';
import LoadingBar from './components/LoadingBar';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setAnswer('');
    setIsLoading(true);
    
    const source = new EventSource(`http://localhost:8000/stream?query=${encodeURIComponent(query)}`);

    source.onmessage = (event) => {
      if (event.data === '[DONE]') {
        source.close();
        setIsLoading(false);
      } else {
        setAnswer(prev => prev + event.data + "\n");
      }
    };

    source.onerror = () => {
      source.close();
      setIsLoading(false);
    };
  };

  return (
    <div className="chat-container">
      <h1 className="chat-title">RAG Chatbot</h1>
      <div className="chat-box">
        <form onSubmit={handleSubmit} className="chat-form">
          <input 
            type="text" 
            placeholder="Ask a question..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="chat-input"
          />
          <button type="submit" className="chat-button">Send</button>
        </form>
        
        <LoadingBar isLoading={isLoading} />
        
        {answer && (
          <div className="answer-container">
            <div className="answer-box">
              {answer}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;