
---

# Music Composition Advisor - Backend

This is the backend API service for the **Music Composition Advisor** project. The backend handles the processing and analysis of song lyrics and genre data, predicting whether the sentiment of a song is positive or negative using a trained AI model. It also manages requests from the frontend and sends back prediction results.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)

---

## Features

- **Lyrics and Genre Analysis**: Receives lyrics and genre as inputs and predicts sentiment.
- **AI Model Integration**: Connects with a sentiment analysis model (e.g., BERT or LSTM-based model) for predictions.
- **Simple API Design**: Provides a RESTful API that can be easily accessed by the frontend.
- **Error Handling**: Ensures proper error messages for invalid requests.

---

## Architecture

The backend is structured to handle requests from the frontend, process the data using a sentiment analysis model, and return the prediction to the frontend.

### Architecture Diagram

![Backend Architecture](..\AI_high_level_architecture.png)

1. **Frontend** sends user input (lyrics and genre) via API requests to the backend.
2. **Backend API** receives the input and forwards it to the **Model Inference Service**.
3. **Model Inference** uses a trained model to predict sentiment based on the input.
4. **Backend API** sends the prediction result back to the frontend.

---

## Setup Instructions

To set up and run the backend locally, follow these steps:

### Prerequisites

- Node.js (version 14 or higher)
- NPM (Node Package Manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd music-composition-advisor-backend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   - Create a `.env` file in the root directory and add the following environment variables:

     ```env
     PORT=5000
     MODEL_PATH=./path/to/your/model
     ```

   - Adjust `MODEL_PATH` based on the location of your trained sentiment model.

4. **Start the server**:
   ```bash
   npm start
   ```

5. **Server Access**:
   - The backend server will run at `http://localhost:5000` by default (if `PORT` is not changed).

---

## API Endpoints

### POST /api/analyze

- **Description**: Receives lyrics and genre data from the frontend and returns the sentiment prediction.
- **URL**: `/api/analyze`
- **Method**: `POST`
- **Request Body**:
  - `lyrics`: (string) The song lyrics to be analyzed.
  - `genre`: (string) The genre of the song.
  
  ```json
  {
    "lyrics": "Sample lyrics text...",
    "genre": "Pop"
  }
  ```

- **Response**:
  - `sentiment`: (string) Predicted sentiment, either "positive" or "negative".

  ```json
  {
    "sentiment": "positive"
  }
  ```

- **Errors**:
  - `400 Bad Request`: Missing or invalid lyrics/genre data.
  - `500 Internal Server Error`: If model prediction fails.

---

## Technologies Used

- **Node.js**: JavaScript runtime for server-side logic.
- **Express**: Web framework for creating API endpoints.
- **TensorFlow.js / Python API**: For loading and running the sentiment analysis model.
- **Dotenv**: For managing environment variables.

---

## Contributing

To contribute:

1. Fork the repository.
2. Create a new branch (`feature/your-feature-name`).
3. Make your changes and commit them.
4. Open a pull request to the main branch.

---