import React, { useState } from 'react';

const GenreSelect = ({ genre, setGenre }) => {
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const genreOptions = [
    'Reggae', 'Art rock', 'New jack swing', 'Adult Contemporary', 'Nigerian Alté', 'Pop', 'Afroswing',
    'R&B/Soul', 'House', 'Latin Urbano', 'Indie/Underground Hip-Hop', 'Emo rap', 'Cloud rap', 'Rap music',
    'Post-disco', 'Dance-pop', 'Indie rock', 'Folk', 'New wave', 'Alternative rock', 'Vocal/Easy Listening',
    'Neo-Soul', 'Alternative R&B', 'Classical', 'Dirty rap', 'Eurodance', 'Regional Mexican', 'House music',
    'Latin trap', 'Christian', 'Dance music', 'Russian Indie', 'Regional Brazilian', 'Rap Québ', 'Electronic dance music',
    'UK R&B', 'Comedy hip hop', 'Sandungueo', 'Disco', 'East Coast Hip-Hop', 'Folk-pop', 'Progressive house', 'Rock',
    'Spoken Word', 'Electro', 'Space rock', 'Punk', 'Electro house', 'Moombahton', 'Afrobeat', 'Ballad', 'Folktronica',
    'Country music', 'Groove metal', 'Arena rock', 'Corrido', 'Cumbia', 'Metal', 'Nigerian Hip Hop', 'Emo', 'Grunge',
    'Alternative hip hop', 'Neo soul', 'Country rap', 'Pop rock', 'Alternative music', 'Doo-wop', 'Synth-pop',
    'Singer-Songwriter', 'Dance/Electronic', 'French Indie', 'K-pop', 'Pop rap', 'Trap metal', 'EDM trap music',
    'Electronic rock', 'Funk', 'Rap rock', 'German Hip Hop', 'Classic Rock', 'Country pop', 'Synthwave', 'MORE',
    'Drill music', 'Electronic music', 'Afropop', 'Dance Pop', 'Hip hop', 'Classic Soul', 'Symphonic rock',
    'Korean Dance', 'Power pop', 'Blue-eyed soul', 'Reggaeton', 'Soft rock', 'Folk rock', 'Teen pop', "Children's Music",
    'Pop soul', 'Hip hop music', 'Rhythm and blues', 'Indie pop', 'pop soul', 'Rock en Español', 'Nu-disco', 'Downtempo',
    'Indian Pop', 'Dream pop', 'Nigerian R&B', 'Contemporary R&B', 'UK Rap', 'emo pop', 'Indian Indie', 'Electropop',
    'Funk rock', 'Reggae en Español', 'Bhangra', 'Pop funk', 'Merengue music', 'Psychedelic pop', 'Soul music',
    'Hip-hop soul', 'Reggae fusion', 'Ambient', 'Hip-Hop/Rap', 'Pop music', 'Tropical house', 'Punk rock', 'Diva house',
    'Mambo', 'Southern Hip-Hop', 'Alternative/Indie', 'Latin music', 'Dancehall', 'Dembow', 'German Pop', 'Country Trap',
    'Bubblegum music', 'Country', 'Afropop', 'Electronica', 'Dubstep', 'Pop-punk', 'Halloween music', 'Latin pop',
    'Volksmusik', 'Lo-fi', 'Bounce music', 'K-Pop', 'Trap music', 'MPB', 'Progressive rap', 'Holiday', 'Art Pop',
    'Acoustic music', 'bedroom pop'
  ];

  const handleAddGenre = (selectedGenre) => {
    if (selectedGenre && !selectedGenres.includes(selectedGenre)) {
      setSelectedGenres([...selectedGenres, selectedGenre]);
      setGenre(''); // Clear the genre input after adding
    }
  };

  const handleRemoveGenre = (genreToRemove) => {
    setSelectedGenres(selectedGenres.filter((g) => g !== genreToRemove));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddGenre(genre); // Add genre when Enter is pressed
    }
  };

  // Filter the genre options based on the input value
  const filteredGenres = genreOptions.filter((g) =>
    g.toLowerCase().includes(genre.toLowerCase())
  );

  return (
    <div className="genre-select">
      <input
        type="text"
        value={genre}
        onChange={(e) => {
          setGenre(e.target.value);
          setShowSuggestions(true); // Show suggestions when user types
        }}
        onKeyPress={handleKeyPress} // Handle enter key press
        placeholder="Search and add genre"
      />
      <button onClick={() => handleAddGenre(genre)}>Enter</button>
      {showSuggestions && genre && (
        <div className="genre-suggestions">
          {filteredGenres.map((g, index) => (
            <div
              key={index}
              className="suggestion-item"
              onClick={() => {
                handleAddGenre(g); // Add genre when clicked
                setShowSuggestions(false); // Hide suggestions after selection
              }}
            >
              {g}
            </div>
          ))}
        </div>
      )}
      <div className="selected-genres">
        {selectedGenres.map((g, index) => (
          <span key={index} className="genre-tag">
            {g} <button onClick={() => handleRemoveGenre(g)} className="remove-btn">x</button>
          </span>
        ))}
      </div>
    </div>
  );
};

export default GenreSelect;
