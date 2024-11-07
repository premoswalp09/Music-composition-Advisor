import React, { useState } from 'react';
import axios from 'axios';
import LyricsInput from './components/LyricsInput';
import GenreSelect from './components/GenreSelect';
import './App.css';

function App() {
  const [lyrics, setLyrics] = useState('');
  const [genre, setGenre] = useState('');
  const [result, setResult] = useState(null);
  const [darkMode, setDarkMode] = useState(false);

  const handleToggleTheme = () => setDarkMode(!darkMode);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/api/predict', { lyrics, genre });
      setResult(response.data.prediction);
    } catch (error) {
      console.error('Error fetching prediction:', error);
    }
  };

  // Determine the result class based on the prediction value
  const resultClass = result === 'Positive' ? 'result positive' : result === 'Negative' ? 'result negative' : 'result neutral';

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      <button className="toggle-button" onClick={handleToggleTheme}>
        {darkMode ? 'Light Mode' : 'Dark Mode'}
      </button>
      <h1>Music Composition Advisor</h1>
      <form onSubmit={handleSubmit}>
        <LyricsInput lyrics={lyrics} setLyrics={setLyrics} />
        <GenreSelect genre={genre} setGenre={setGenre} />
        <button type="submit" className="submit-button">Analyze</button>
      </form>
      {result && <div className={resultClass}>Prediction: {result}</div>}
      {lyrics && (
        <div className="lyrics-display">
          <h3>Entered Lyrics:</h3>
          <div className="lyrics-box">{lyrics}</div>
        </div>
      )}
    </div>
  );
}

export default App;
