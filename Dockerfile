FROM python:3.10-bullseye

WORKDIR /app
# 해당 디렉토리에 있는 모든 하위항목들을 현재 디렉토리로 복사 ,, 여기서 지정
COPY . .

# 필요한 패키지 및 의존성 파일 설치
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
CMD ["-h"]
