#!/bin/bash

GEMINI_API_KEY="AIzaSyBQt3EcF_c8_ttr86p3H9P8YgSH8yjDp_g"

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" \
  -H 'Content-Type: application/json' \
  -H 'X-goog-api-key: AIzaSyBQt3EcF_c8_ttr86p3H9P8YgSH8yjDp_g' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "Explain how AI works in a few words"
          }
        ]
      }
    ]
  }'