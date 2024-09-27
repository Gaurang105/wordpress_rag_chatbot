# Backend Usage Documentation for PHP Frontend

## Overview

This document outlines how to interact with the FastAPI-based backend for the Rajesh Jain blog chatbot from a PHP frontend. The backend provides a RESTful API that allows querying the chatbot with natural language questions.

## To Run

Go to backend folder: cd backend

Then create a environment variable and activate the environment variable and then run this command: pip install -r requirements.txt

Then write this command to start the server: python main.py

## Base URL

The base URL for all API endpoints is:

```
http://[your-server-ip]:8000
```

Replace `[your-server-ip]` with the actual IP address or domain name where the backend is hosted.

## API Endpoints

### 1. Health Check

- **Endpoint**: `/health`
- **Method**: GET
- **Description**: Check if the backend server is running and ready to accept requests.
- **Response**: 
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Query the Chatbot

- **Endpoint**: `/query`
- **Method**: POST
- **Description**: Send a question to the chatbot and receive a response.
- **Request Body**:
  ```json
  {
    "query": "Your question here"
  }
  ```
- **Response**:
  ```json
  {
    "response": "Chatbot's answer here"
  }
  ```

## Using the API with PHP

Here's an example of how to use the API from your PHP frontend:

```php
<?php

function queryBackend($question) {
    $url = 'http://[your-server-ip]:8000/query';
    $data = array('query' => $question);

    $options = array(
        'http' => array(
            'header'  => "Content-type: application/json\r\n",
            'method'  => 'POST',
            'content' => json_encode($data)
        )
    );

    $context  = stream_context_create($options);
    $result = file_get_contents($url, false, $context);

    if ($result === FALSE) {
        return "Error connecting to the backend.";
    }

    $response = json_decode($result, true);
    return $response['response'];
}

// Example usage
$question = "What does Rajesh Jain say about digital transformation?";
$answer = queryBackend($question);
echo $answer;

?>
```

## Error Handling

The backend may return error responses in the following format:

```json
{
  "error": "Error message here"
}
```