import React, { useState, useEffect } from 'react';
import { getProfile, updateProfile } from '../api';

function Profile({ token }) {
  const [diet, setDiet] = useState('');

  useEffect(() => {
    (async () => {
      const res = await getProfile(token);
      setDiet(res.data.dietary_preferences);
    })();
  }, [token]);

  const handleSave = async (e) => {
    e.preventDefault();
    await updateProfile(token, diet);
    alert('Profile updated');
  };

  return (
    <div>
      <h2>Profile</h2>
      <form onSubmit={handleSave}>
        <textarea value={diet} onChange={e=>setDiet(e.target.value)} placeholder="Dietary preferences" />
        <button>Save</button>
      </form>
    </div>
  );
}

export default Profile;
