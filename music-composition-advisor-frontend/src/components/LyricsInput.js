import React from 'react';

function LyricsInput({ lyrics, setLyrics }) {
  return (
    <div>
      <label>Enter Lyrics:</label>
      <textarea
        value={lyrics}
        onChange={(e) => setLyrics(e.target.value)}
        rows="5"
        placeholder="Type the lyrics here..."
      />
    </div>
  );
}

export default LyricsInput;
