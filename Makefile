IMAGE_NAME ?= "opsani/servox:latest"

ifneq (,$(wildcard ./.env))
    include .env
    export
    ENV_FILE_PARAM = --env-file .env
endif
OPSANI_TOKEN_FILE ?= "/dev/null"

.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:latest .

.PHONY: config
config:
	@$(MAKE) -e SERVO_ARGS="servo generate --force" run

.PHONY: run
run: build
	docker run -it \
		-v $(CURDIR)/servo.yaml:/servo/servo.yaml \
		-v $(CURDIR)/.env:/root/.env:ro \
		-v ${HOME}/.kube:/root/.kube:ro \
		-v ${HOME}/.aws:/root/.aws:ro 	\
		-v ${OPSANI_TOKEN_FILE:-/dev/null}:/servo/opsani.token \
		$(ENV_FILE_PARAM) \
		$(IMAGE_NAME) \
		$(SERVO_ARGS)

.PHONY: push
push: build
	docker push ${IMAGE_NAME}

.PHONY: format
format:
	poetry run isort .
	poetry run autoflake --recursive --remove-all-unused-imports --remove-unused-variables --in-place servo tests
	poetry run black servo tests

.PHONY: lint
lint:
	poetry run flake8 servo
	poetry run mypy servo
	poetry run black --check servo --diff
	poetry run isort --recursive --check-only servo
