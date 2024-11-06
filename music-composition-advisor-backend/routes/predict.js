const express = require('express');
const router = express.Router();
const axios = require('axios');
const { spawn } = require('child_process');

router.post('/', (req, res) => {
  const { lyrics, genre } = req.body;

  // Spawn a Python process to run the sentiment model script
  const pythonProcess = spawn('python', ['./python/sentiment_model.py', lyrics, genre]);

  pythonProcess.stdout.on('data', (data) => {
    const prediction = data.toString().trim();
    res.json({ prediction });
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Error: ${data}`);
    res.status(500).json({ error: 'Prediction failed' });
  });
});

module.exports = router;
