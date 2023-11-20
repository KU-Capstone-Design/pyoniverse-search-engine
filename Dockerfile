# slim-bullseye. 45.22MB
# alpine은 tensorflow를 설치하는 과정이 복잡하기 때문에 bullseye 선택
FROM python:3.10-slim-bullseye AS builder

WORKDIR /app
COPY . .
RUN apt-get update && apt-get upgrade && apt-get install locales git -y && locale-gen en_US.UTF-8 &&  \
    python -m pip install --upgrade pip
# Install ML Base
RUN pip install --no-cache-dir tensorflow sentence-transformers rank_bm25
RUN pip install --no-cache-dir --no-deps git+https://github.com/haven-jeon/PyKoSpacing.git # 4.87MB
# Install API Base
RUN pip install --no-cache-dir -r requirements-docker.txt # 21.1MB

# 2.13GB로 작아짐
FROM python:3.10-slim-bullseye AS deployer

WORKDIR /app
# 해당 디렉토리에 있는 모든 하위항목들을 현재 디렉토리로 복사 ,, 여기서 지정
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
RUN apt-get update && apt-get install curl -y && chmod +x ./embedding.sh && ./embedding.sh && apt-get remove curl -y  \
    && rm -r resource

CMD ["uvicorn", "main:app", "--port=7777", "--host=0.0.0.0"]
#ENTRYPOINT ["bash"]
