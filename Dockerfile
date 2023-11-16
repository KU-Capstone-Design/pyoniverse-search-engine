# slim-bullseye. 45.22MB
# alpine은 tensorflow를 설치하는 과정이 복잡하기 때문에 bullseye 선택
FROM python:3.10-slim-bullseye AS builder

COPY requirements-docker.txt .
RUN apt-get update && apt-get upgrade
# Locale 설정 및 pip upgrade
RUN apt-get install locales git -y
RUN locale-gen en_US.UTF-8
RUN python -m pip install --upgrade pip

# Install ML Base
RUN pip install --no-cache-dir tensorflow # 1.52GB
RUN pip install --no-cache-dir --no-deps git+https://github.com/haven-jeon/PyKoSpacing.git # 4.87MB
RUN pip install --no-cache-dir -U sentence-transformers # 907MB
RUN pip install --no-cache-dir rank_bm25 # 45.5KB
# Install API Base
RUN pip install --no-cache-dir -r requirements-docker.txt # 21.1MB

# 2.13GB로 작아짐
FROM python:3.10-slim-bullseye AS deployer

WORKDIR /app
# 해당 디렉토리에 있는 모든 하위항목들을 현재 디렉토리로 복사 ,, 여기서 지정
COPY lib lib
COPY main.py .
COPY .env .
COPY setup.sh .
COPY embedding.sh .
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN apt-get update && apt-get install curl -y
RUN chmod +x ./setup.sh && ./setup.sh
RUN chmod +x ./embedding.sh && ./embedding.sh
RUN apt-get remove curl -y && apt-get clean
RUN rm setup.sh embedding.sh

ENTRYPOINT ["uvicorn", "main:app"]
CMD ["--port=7777", "--host=0.0.0.0"]
