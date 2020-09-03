FROM peterevans/vegeta AS vegeta
FROM python:3.8-slim

ARG SERVO_ENV=development

ENV SERVO_ENV=${SERVO_ENV} \    
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

RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && apt-get purge -y --auto-remove \
  && rm -rf /var/lib/apt/lists/*

# Install Vegeta
COPY --from=vegeta /bin/vegeta /bin/vegeta

# Build Servo
WORKDIR /servo

# The entry point is copied in ahead of the main sources
# so that the servo CLI is installed by Poetry. The sequencing
# here accelerates builds by ensuring that only essential
# cache friendly files are in the stage when Poetry executes.
COPY poetry.lock pyproject.toml ./
COPY servo/entry_points.py servo/entry_points.py

RUN pip install poetry==1.0.* \
  && poetry install \
    $(if [ "$SERVO_ENV" = 'production' ]; then echo '--no-dev'; fi) \
    --no-interaction \
  # Clean poetry cache for production
  && if [ "$SERVO_ENV" = 'production' ]; then rm -rf "$POETRY_CACHE_DIR"; fi

# Add common connectors distributed as standalone libraries
RUN poetry add servo-webhooks

# Copy the servo sources
COPY . ./

ENTRYPOINT [ "/servo/entrypoint.sh" ]
CMD [ "run" ]
