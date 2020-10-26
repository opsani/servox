IMAGE_NAME ?= "opsani/servox:edge"

ifneq (,$(wildcard ./.env))
    include .env
    export
    ENV_FILE_PARAM = --env-file .env
endif
OPSANI_TOKEN_FILE ?= "/dev/null"

.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:latest .

.PHONY: generate
generate:
	@$(MAKE) -e SERVO_ARGS="generate --force" run

.PHONY: config
config:
	@$(MAKE) -e SERVO_ARGS="config" run

.PHONY: run
run: build
	docker run -it \
		-v $(CURDIR)/servo.yaml:/servo/servo.yaml \
		-v ${HOME}/.kube:/root/.kube:ro \
		-v ${HOME}/.aws:/root/.aws:ro 	\
		-v ${OPSANI_TOKEN_FILE:-/dev/null}:/servo/opsani.token \
		$(ENV_FILE_PARAM) \
		$(IMAGE_NAME) \
		${SERVO_ARGS:-run}

.PHONY: push
push: build
	docker push ${IMAGE_NAME}

.PHONY: format
format:
	poetry run isort .
	poetry run autoflake --recursive \
		--ignore-init-module-imports \
		--remove-all-unused-imports  \
		--remove-unused-variables    \
		--in-place servo tests

.PHONY: typecheck
typecheck:
	poetry run mypy servo || true

lint-docs:
	poetry run flake8-markdown "**/*.md" || true

.PHONY: lint
lint: typecheck
	poetry run flakehell lint --count

.PHONY: kubeconfig
kubeconfig:
	kubectl config view \
    	--minify --flatten \
		--context=servox-integration-tests > $(CURDIR)/tests/kubeconfig

.PHONY: test
test:
	poetry run pytest --cov=servo --cov-report=term-missing:skip-covered --cov-config=setup.cfg tests

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --hook-stage manual --all-files
