import React from 'react';
import AudioAnalyzer from './components/AudioAnalyzer';
import LyricsAnalyzer from './components/LyricsAnalyzer';
import { Container, Typography } from '@mui/material';

const App = () => {
  return (
    <Container maxWidth="sm" sx={{ marginTop: 4 }}>
      <Typography variant="h4" align="center" gutterBottom>
        Music Sentiment Analyzer
      </Typography>
      <AudioAnalyzer />
    </Container>
  );
};

export default App;
