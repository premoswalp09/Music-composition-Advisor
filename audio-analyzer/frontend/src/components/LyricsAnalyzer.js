import React, { useState } from 'react';
import axios from 'axios';
import { Box, Button, TextField, Typography, Card, CardContent } from '@mui/material';

const LyricsAnalyzer = () => {
    const [lyrics, setLyrics] = useState('');
    const [sentiment, setSentiment] = useState('');

    const handleLyricsChange = (event) => {
        setLyrics(event.target.value);
    };

    const handleLyricsSubmit = async (event) => {
        event.preventDefault();

        try {
            const response = await axios.post('http://localhost:5000/api/analyze-lyrics', { lyrics });
            setSentiment(response.data.sentiment);
        } catch (error) {
            console.error('Error analyzing lyrics:', error);
            setSentiment('Error analyzing lyrics.');
        }
    };

    return (
        <Box sx={{ padding: 2, border: '1px solid #ccc', borderRadius: 2 }}>
            <Typography variant="h5">Lyrics Sentiment Analyzer</Typography>
            <form onSubmit={handleLyricsSubmit}>
                <TextField
                    value={lyrics}
                    onChange={handleLyricsChange}
                    placeholder="Enter song lyrics here"
                    required
                    multiline
                    rows={4}
                    fullWidth
                    margin="normal"
                />
                <Button variant="contained" type="submit" color="primary">
                    Analyze
                </Button>
            </form>
            {sentiment && (
                <Card sx={{ marginTop: 2 }}>
                    <CardContent>
                        <Typography variant="h6" color="text.primary">
                            Sentiment Result
                        </Typography>
                        <Typography variant="body1" sx={{ fontWeight: 'bold', color: sentiment === 'Positive' ? 'green' : 'red' }}>
                            {sentiment}
                        </Typography>
                    </CardContent>
                </Card>
            )}
        </Box>
    );
};

export default LyricsAnalyzer;
