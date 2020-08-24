FROM python:3.8

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
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    # Vegeta
    VEGETA_VERSION=12.8.3

# Install kubectl
RUN apt-get update \
  && apt-get install -y apt-utils apt-transport-https gnupg2 \
  && curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | tee -a /etc/apt/sources.list.d/kubernetes.list \
  && apt-get update \
  && apt-get install -y kubectl

# Install AWS IAM Authenticator
RUN curl -o /usr/local/bin/aws-iam-authenticator https://amazon-eks.s3.us-west-2.amazonaws.com/1.17.7/2020-07-08/bin/linux/amd64/aws-iam-authenticator \
  && chmod +x /usr/local/bin/aws-iam-authenticator

# Install Vegeta
RUN wget -q "https://github.com/tsenart/vegeta/releases/download/v$VEGETA_VERSION/vegeta-$VEGETA_VERSION-linux-amd64.tar.gz" -O /tmp/vegeta.tar.gz \
 && cd bin \
 && tar xzf /tmp/vegeta.tar.gz \
 && rm /tmp/vegeta.tar.gz

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
