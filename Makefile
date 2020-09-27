IMAGE_NAME ?= "opsani/servox:latest"

ifneq (,$(wildcard ./.env))
    include .env
    export
    ENV_FILE_PARAM = --env-file .env
endif
OPSANI_TOKEN_FILE ?= "/dev/null"

.PHONY: build
build:
	docker build -t ${IMAGE_NAME} .

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
