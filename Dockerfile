FROM python:3.6-buster

WORKDIR /usr/src/app

RUN apt update && apt install -y libsndfile1 ffmpeg

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000

CMD [ "python", "./main.py" ]