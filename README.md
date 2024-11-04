# Music Composition Advisor

## Project Overview
The Music Composition Advisor is an AI-driven project aimed at training a model to analyze song lyrics and genres to predict the positive or negative impact of music. This project leverages machine learning techniques to provide insights into how lyrics and genre influence listener emotions.

## Features
- Lyrics analysis and genre classification.
- Predictive modeling for determining the emotional impact of music.
- User-friendly interface for interaction and feedback.

## Architecture
The architecture of the Music Composition Advisor is designed to facilitate seamless data flow from user input through to model inference and output. Below is a high-level view of the architecture:

![Architecture Diagram](AI_high_level_architecture.png)

### Components
- **User Interface**: The front-end where users can input lyrics and select genres.
- **Frontend**: Built using React to handle user interactions and API requests.
- **Backend**: A server that processes requests, manages data, and interfaces with the machine learning model.
- **Model Inference**: The core component that applies trained models (like BERT or LSTM) to analyze input and provide predictions.
- **Data Collection and Preprocessing**: Handles the collection of lyrics data and prepares it for model training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music-composition-advisor.git
   cd music-composition-advisor


2. Install the required dependencies:
   ```bash
   npm install
   ```

3. Set up your environment:
   - Ensure you have Node.js and Python installed.
   - Create a `.env` file for environment variables as needed.

## Usage
1. Start the backend server:
   ```bash
   node backend/server.js
   ```

2. Run the frontend application:
   ```bash
   npm start
   ```

3. Open your browser and go to `http://localhost:3000` to access the Music Composition Advisor.

## Contributing
We welcome contributions! Please fork the repository and submit a pull request with your changes.

## Acknowledgements
- Thanks to all contributors for their valuable input.
- Special thanks to the resources that provided guidance on machine learning and web development.