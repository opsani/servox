IMAGE_NAME ?= "opsani/servox:edge"
KUBETEST_CONTEXT ?= "kubetest"

ifneq (,$(wildcard ./.env))
    include .env
    export
    ENV_FILE_PARAM = --env-file .env
endif
OPSANI_TOKEN_FILE ?= "/dev/null"

define vscode_settings
{
    "python.pythonPath": "$(shell poetry env info -p)/bin/python",
    "terminal.integrated.shellArgs.linux": ["poetry shell"],
    "terminal.integrated.shellArgs.osx": ["poetry shell"],
    "files.exclude": {
        "**/.git": true,
        "**/.DS_Store": true,
        "**/*.pyc": true,
        "**/__pycache__": true,
        "**/.mypy_cache": true
    },
    "python.linting.enabled": true
}
endef
export vscode_settings

.PHONY: init
init:
	mkdir -p .vscode
	touch .vscode/settings.json
	@echo "$$vscode_settings" > .vscode/settings.json
	poetry install
	poetry run servo init

.PHONY: vscode
vscode:
	source "$(shell poetry env info -p)/bin/activate" --prompt "poetry env"
	code .

.PHONY: build
build:
	DOCKER_BUILDKIT=1 docker build -t ${IMAGE_NAME} --build-arg BUILDKIT_INLINE_CACHE=1 --cache-from opsani/servox:edge .

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

.PHONY: lint-docs
lint-docs:
	poetry run flake8-markdown "**/*.md" || true

.PHONY: lint
lint: typecheck
	poetry run flakehell lint --count

.PHONY: scan
scan:
	poetry run bandit -r servo

.PHONY: test-kubeconfig
test-kubeconfig:
	@kubectl config view \
    	--minify --flatten \
		> $(CURDIR)/tests/kubeconfig
	@kubectl config rename-context \
		--kubeconfig=$(CURDIR)/tests/kubeconfig \
		$(shell kubectl config current-context) \
		$(KUBETEST_CONTEXT)
	@echo "Saved current kubeconfig context '$(shell kubectl config current-context)' as '$(KUBETEST_CONTEXT)' in tests/kubeconfig"

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --hook-stage manual --all-files

.PHONY: autotest
autotest:
	poetry run watchgod autotest.main

.PHONY: test
test:
	poetry run pytest -n auto --dist loadscope

.PHONY: test-coverage
	poetry run pytest --cov=servo --cov-report=term-missing:skip-covered --cov-config=setup.cfg

.PHONY: test-unit
test-unit:
	poetry run pytest -T unit -n auto --dist loadscope

.PHONY: test-integration
test-integration:
	poetry run pytest -T integration -n auto --dist loadscope

.PHONY: test-system
test-system:
	poetry run pytest -T system -n auto --dist loadscope
