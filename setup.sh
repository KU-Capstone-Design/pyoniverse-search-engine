#!/bin/bash

uvicorn --port=7777 main:app&
sleep 10

# Model Download & Embedding 실시
URL=http://localhost:7777/ping
MAX_RETRIES=10  # 최대 재시도 횟수

retries=0

while true; do
    response=$(curl -X GET --max-time 1800 -s -o /dev/null -w "%{http_code}" $URL)

    if [ "$response" -eq 200 ]; then
        echo "OK"
        break
    else
        echo "False: $response"

        retries=$((retries + 1))

        if [ $retries -eq $MAX_RETRIES ]; then
            echo "Exceed ($MAX_RETRIES). End script"
            exit 1
        fi
        sleep 30
        echo "Retry ($retries/$MAX_RETRIES)"
    fi
done
