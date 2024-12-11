import React, { useState } from 'react';
import { askQuestion } from '../api';

function Ask({ token }) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');

  const handleAsk = async (e) => {
    e.preventDefault();
    const res = await askQuestion(token, query);
    setAnswer(res.data.answer);
  };

  return (
    <div>
      <h2>Ask a question</h2>
      <form onSubmit={handleAsk}>
        <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Your question" />
        <button>Ask</button>
      </form>
      {answer && <p><strong>Answer:</strong> {answer}</p>}
    </div>
  );
}

export default Ask;
