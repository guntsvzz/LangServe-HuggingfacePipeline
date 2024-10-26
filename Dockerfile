FROM python:3.10.9

COPY . .

WORKDIR /


RUN pip install poetry==1.6.1
RUN poetry config virtualenvs.create false

RUN poetry install  --no-interaction --no-ansi --no-root

EXPOSE 8080


# Clear apt for optimizing image size
RUN apt clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]