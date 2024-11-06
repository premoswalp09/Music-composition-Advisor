import React from 'react';

function GenreSelect({ genre, setGenre }) {
  return (
    <div>
      <label>Select Genre:</label>
      <select value={genre} onChange={(e) => setGenre(e.target.value)}>
        <option value="">Select Genre</option>
        <option value="pop">Pop</option>
        <option value="rock">Rock</option>
        <option value="jazz">Jazz</option>
        <option value="classical">Classical</option>
        <option value="hip-hop">Hip-Hop</option>
        {/* Add other genres as needed */}
      </select>
    </div>
  );
}

export default GenreSelect;
