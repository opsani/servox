FROM python:3.8

ENV PYTHONUNBUFFERED 1
ENV OPSANI_OPTIMIZER example.com/app-name
ENV OPSANI_TOKEN_FILE /var/run/secrets/opsani.com/optimizer.token
ENV OPSANI_CONFIG_FILE /servo/servo.yaml

WORKDIR /servo

COPY poetry.lock pyproject.toml ./
RUN pip install poetry==1.0.* && \
    poetry config virtualenvs.create false && \
    poetry install

COPY . ./

CMD poetry run servo \
    --optimizer ${OPSANI_OPTIMIZER} \
    --token-file ${OPSANI_TOKEN_FILE} \
    --config-file ${OPSANI_CONFIG_FILE} \
    run
