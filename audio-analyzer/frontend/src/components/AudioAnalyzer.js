import React, { useState } from 'react';  
import axios from 'axios';  
import {   
    Box,   
    Button,   
    TextField,   
    Typography,   
    Card,   
    CardContent,  
    CircularProgress,  
    List,  
    ListItem,  
    ListItemText,  
    Alert,  
    Paper  
} from '@mui/material';  

const AudioAnalyzer = () => {  
    const [file, setFile] = useState(null);  
    const [loading, setLoading] = useState(false);  
    const [error, setError] = useState(null);  
    const [analysisResult, setAnalysisResult] = useState(null);  

    const handleFileChange = (event) => {  
        const selectedFile = event.target.files[0];  
        if (selectedFile) {  
            // Specific MP3 validation  
            const isMP3 = selectedFile.type === 'audio/mpeg' ||   
                         selectedFile.type === 'audio/mp3' ||   
                         selectedFile.name.toLowerCase().endsWith('.mp3');  
                         
            if (!isMP3) {  
                setError('Please select an MP3 file only');  
                setFile(null);  
                return;  
            }  
            
            if (selectedFile.size > 10 * 1024 * 1024) {  
                setError('MP3 file size should be less than 10MB');  
                setFile(null);  
                return;  
            }  
            
            setFile(selectedFile);  
            setError(null);  
        } else {  
            setError('Please select an MP3 file');  
            setFile(null);  
        }  
    };  

    const handleFileSubmit = async (event) => {  
        event.preventDefault();  
        if (!file) {  
            setError('Please select an MP3 file first');  
            return;  
        }  

        setLoading(true);  
        setError(null);  
        setAnalysisResult(null);  

        const formData = new FormData();  
        formData.append('audio', file);  

        try {  
            const response = await axios.post('http://localhost:5000/api/analyze-audio', formData, {  
                headers: {  
                    'Content-Type': 'multipart/form-data'  
                }  
            });  
            setAnalysisResult(response.data);  
        } catch (error) {  
            console.error('Error analyzing audio:', error);  
            setError(error.response?.data?.error || 'Error analyzing the MP3 file. Please try again.');  
        } finally {  
            setLoading(false);  
        }  
    };  

    const formatProbability = (prob) => `${(prob * 100).toFixed(1)}%`;  

    const getConfidenceColor = (confidence) => {  
        if (confidence >= 0.7) return 'success.main';  
        if (confidence >= 0.4) return 'warning.main';  
        return 'error.main';  
    };  

    return (  
        <Box sx={{   
            padding: 3,   
            border: '1px solid #ccc',   
            borderRadius: 2,   
            marginBottom: 2,  
            maxWidth: 600,  
            mx: 'auto',  
            backgroundColor: '#fff',  
            boxShadow: 1  
        }}>  
            <Typography variant="h5" color="primary.main" sx={{ mb: 3 }}>  
                MP3 Genre Analyzer  
            </Typography>  
            
            <form onSubmit={handleFileSubmit}>  
                <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>  
                    <Typography variant="body2" color="text.secondary" gutterBottom>  
                        Upload your MP3 file (max 10MB)  
                    </Typography>  
                    
                    <TextField  
                        type="file"  
                        onChange={handleFileChange}  
                        accept=".mp3,audio/mpeg"  
                        required  
                        fullWidth  
                        margin="normal"  
                        error={!!error}  
                        helperText={error}  
                        InputProps={{  
                            inputProps: {  
                                accept: '.mp3,audio/mpeg'  
                            }  
                        }}  
                    />  

                    {file && (  
                        <Box sx={{   
                            mt: 1,   
                            p: 1,   
                            bgcolor: 'background.default',  
                            borderRadius: 1  
                        }}>  
                            <Typography variant="body2">  
                                Selected: {file.name}  
                            </Typography>  
                        </Box>  
                    )}  
                </Paper>  

                <Button   
                    variant="contained"   
                    type="submit"   
                    color="primary"  
                    disabled={!file || loading}  
                    fullWidth  
                    sx={{ height: 48 }}  
                >  
                    {loading ? (  
                        <>  
                            <CircularProgress size={24} sx={{ mr: 1, color: 'white' }} />  
                            Analyzing MP3...  
                        </>  
                    ) : 'Analyze MP3'}  
                </Button>  
            </form>  

            {error && (  
                <Alert severity="error" sx={{ mt: 2 }}>  
                    {error}  
                </Alert>  
            )}  

            {analysisResult && (  
                <Card sx={{ mt: 3, backgroundColor: 'background.paper' }}>  
                    <CardContent>  
                        <Typography variant="h6" gutterBottom color="primary">  
                            Analysis Results  
                        </Typography>  

                        <Box sx={{ mb: 3 }}>  
                            <Typography variant="subtitle1" fontWeight="bold">  
                                Predicted Genre:  
                            </Typography>  
                            <Typography   
                                variant="h5"   
                                color={getConfidenceColor(analysisResult.confidence)}  
                                sx={{ my: 1 }}  
                            >  
                                {analysisResult.predicted_genre.toUpperCase()}  
                            </Typography>  
                            <Typography variant="body2" color="text.secondary">  
                                Confidence: {formatProbability(analysisResult.confidence)}  
                            </Typography>  
                        </Box>  

                        <Typography variant="subtitle1" fontWeight="bold" sx={{ mb: 1 }}>  
                            Top 3 Predictions:  
                        </Typography>  
                        <List sx={{ bgcolor: 'background.default', borderRadius: 1 }}>  
                            {analysisResult.top_3_predictions.map((pred, index) => (  
                                <ListItem   
                                    key={index}  
                                    sx={{  
                                        borderBottom: index < 2 ? '1px solid rgba(0,0,0,0.08)' : 'none'  
                                    }}  
                                >  
                                    <ListItemText  
                                        primary={  
                                            <Typography variant="body1" fontWeight={index === 0 ? 'bold' : 'normal'}>  
                                                {`${index + 1}. ${pred.genre.toUpperCase()}`}  
                                            </Typography>  
                                        }  
                                        secondary={`Probability: ${formatProbability(pred.probability)}`}  
                                    />  
                                </ListItem>  
                            ))}  
                        </List>  
                    </CardContent>  
                </Card>  
            )}  
        </Box>  
    );  
};  

export default AudioAnalyzer;