# Rajesh Jain Chatbot API

This document provides information on how to interact with the Rajesh Jain chatbot backend API from PHP frontend.

## API Endpoint

The base URL for the API is: `http://<your-server-ip>:<PORT>`

Replace `<your-server-ip>` with the actual IP address or domain name where the FastAPI server is running, and `<PORT>` with the port number specified in the environment variables (default is 8000).

## Available Endpoints

### 1. Health Check

- **URL:** `/health`
- **Method:** GET
- **Description:** Check if the server is up and running.
- **Response:**
  - Status 200 OK
    ```json
    {
      "status": "ok"
    }
    ```
  - Status 503 Service Unavailable
    ```json
    {
      "detail": "Service is not ready"
    }
    ```

### 2. Query the Chatbot

- **URL:** `/query`
- **Method:** POST
- **Description:** Send a query to the chatbot and receive a response.
- **Request Body:**
  ```json
  {
    "query": "Your question here"
  }
  ```
- **Response:**
  - Status 200 OK
    ```json
    {
      "response": "Chatbot's response here"
    }
    ```

## Usage

To use this API, send HTTP requests to the appropriate endpoints as described above. Make sure to replace `<your-server-ip>` and `<PORT>` with the actual values for your server.

For the query endpoint, send a POST request with a JSON body containing the "query" key and your question as the value.
