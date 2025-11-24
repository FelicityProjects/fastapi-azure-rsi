# Dockerfile

FROM python:3.11-slim

ENV PORT=8000

WORKDIR /app

# requirements.txt 파일의 위치가 이 디렉토리(./)와 일치해야 합니다.
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# main.py 파일의 위치가 이 디렉토리(./)와 일치해야 합니다.
COPY . .

EXPOSE ${PORT}

# CMD 명령의 'main:app'은 main.py 파일의 'app' 인스턴스를 실행함을 의미합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]