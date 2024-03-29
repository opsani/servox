# Copyright 2022 Cisco Systems, Inc. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=alpine:latest

FROM peterevans/vegeta AS vegeta
FROM $BASE_IMAGE

ARG BASE_IMAGE
ARG SERVO_ENV=development

ENV SERVO_ENV=${SERVO_ENV} \
  BASE_IMAGE=${BASE_IMAGE} \
  # Pyenv
  PYENV_ROOT=/home/appdynamics/.pyenv \
  # Python
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  # PIP
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  # Poetry
  POETRY_VIRTUALENVS_CREATE=false

RUN apk -U upgrade && apk add --no-cache curl

RUN if [ "$BASE_IMAGE" = 'alpine:latest' ]; then \
    apk add --no-cache shadow \ 
    && addgroup -S appdynamics \
    && groupmod -g 9001 appdynamics \
    && adduser -S appdynamics -G appdynamics \
    && usermod -u 9001 appdynamics \
    && apk del shadow \
    && mkdir -p /opt/appdynamics \
    && chown -R appdynamics:appdynamics /opt/appdynamics \
    && chmod 777 /var/log; \
  else \
    echo "skipping add user"; \
  fi

# Install Vegeta
COPY --from=vegeta /bin/vegeta /bin/vegeta

# Add kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl \
  && chmod +x ./kubectl \
  && mv ./kubectl /usr/local/bin

# Install pyenv and python
RUN apk add --no-cache bash git

ENV LANG=C.UTF-8 \
  PYENV_ROOT=/home/appdynamics/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/plugins/pyenv-virtualenv/shims:$PYENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN curl -sL https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

RUN apk add --no-cache \
  # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
  git bash build-base libffi-dev openssl-dev bzip2-dev zlib-dev xz-dev readline-dev sqlite-dev tk-dev \
  # Install latest libuv and build uvloop workaround for CVE-2024-24806
  libuv

WORKDIR /servo
COPY .python-version ./

RUN pyenv install --verbose `cat .python-version` && \
  pyenv global `cat .python-version` && \
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/appdynamics/.bashrc && \
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/appdynamics/.bashrc && \ 
  echo 'eval "$(pyenv init - bash)"' >> /home/appdynamics/.bashrc && \ 
  echo 'eval "$(pyenv virtualenv-init - bash)"' >> /home/appdynamics/.bashrc && \
  ln -s /home/appdynamics/.bashrc /home/appdynamics/.profile
RUN chown -R appdynamics:appdynamics $PYENV_ROOT /home/appdynamics/.bashrc /home/appdynamics/.profile
SHELL [ "/bin/bash", "-l", "-c" ]
RUN echo 'eval $(pyenv sh-activate --quiet)' >> /home/appdynamics/.bashrc

USER appdynamics

# Use system installed latest libuv and build uvloop workaround for CVE-2024-24806
# https://github.com/MagicStack/uvloop/issues/589
RUN pip install uvloop==0.18.0 --global-option="--use-system-libuv"

# Build Servo
RUN pip install --upgrade pip setuptools

# The entry point is copied in ahead of the main sources
# so that the servo CLI is installed by Poetry. The sequencing
# here accelerates builds by ensuring that only essential
# cache friendly files are in the stage when Poetry executes.
COPY poetry.lock pyproject.toml README.md CHANGELOG.md ./
COPY servo/entry_points.py servo/entry_points.py

RUN pip install poetry==1.7.0
RUN poetry install --no-dev --no-interaction \
  # Clean poetry cache for production
  && if [ "$SERVO_ENV" = 'production' ]; then rm -rf /home/appdynamics/.cache/pypoetry; fi

USER root
RUN apk del git curl build-base gcc libffi-dev openssl-dev bzip2-dev zlib-dev xz-dev readline-dev sqlite-dev tk-dev
USER appdynamics

# Copy the servo sources
COPY . ./

ENTRYPOINT [ "/servo/entrypoint.sh" ]
CMD [ "run" ]
