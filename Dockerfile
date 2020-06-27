FROM python:3.8

ENV SERVO_ENV=${SERVO_ENV:-development} \    
    # Python
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \    
    # PIP
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # Poetry
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry'    

WORKDIR /servo

COPY poetry.lock pyproject.toml ./
COPY servo/entry_points.py servo/entry_points.py

RUN pip install poetry==1.0.* \
  && poetry install \
    $(if [ "$SERVO_ENV" = 'production' ]; then echo '--no-dev'; fi) \
    --no-interaction \
  # Clean poetry cache for production
  && if [ "$SERVO_ENV" = 'production' ]; then rm -rf "$POETRY_CACHE_DIR"; fi

COPY . ./

# Allow literal or volume mounted tokens
CMD servo \
    --optimizer ${OPSANI_OPTIMIZER:?must be configured} \
    --config-file /servo/servo.yaml \
    $(if [ ! -z ${OPSANI_TOKEN} ]; then \
        echo "--token ${OPSANI_TOKEN}"; \
      else \
        echo "--token-file /var/run/secrets/opsani.com/optimizer.token"; \
      fi) \
    run
