SHELL := /bin/bash
.DEFAULT_GOAL := help
export CHEFNMR2_PROFILE := $(value PROFILE)
export CHEFNMR2_CONTEXT := $(value DOCKER_CONTEXT)
export CHEFNMR2_POOL := $(value WHEELHOUSE)

.PHONY: help sdk/acquire lock acquire build check
help:
	@echo 'make sdk/acquire DOCKER_CONTEXT=science WHEELHOUSE=/cache/pool'
	@echo 'make lock|acquire|build PROFILE=provider|worker-cpu|worker-cuda DOCKER_CONTEXT=science WHEELHOUSE=/cache/pool'
	@echo 'make check DOCKER_CONTEXT=science'
	@echo 'make config|manual/config|manual|status|logs DEPLOYMENT=localhost DOCKER_CONTEXT=science'
	@echo 'make qualify DEPLOYMENT=localhost DOCKER_CONTEXT=science'
	@echo 'make weights/download|weights/verify DOCKER_CONTEXT=science'

sdk/acquire lock acquire build:
	@python3 -B scripts/build.py $@ --profile "$$CHEFNMR2_PROFILE" --context "$$CHEFNMR2_CONTEXT" --pool "$$CHEFNMR2_POOL"

check:
	@python3 -B scripts/runtime.py check --context "$$CHEFNMR2_CONTEXT"

export CHEFNMR2_DEPLOYMENT := $(value DEPLOYMENT)
.PHONY: qualify config manual/config status logs manual weights/download weights/verify
qualify config manual/config status logs manual weights/download weights/verify:
	@python3 -B scripts/runtime.py $@ --context "$$CHEFNMR2_CONTEXT" --deployment "$$CHEFNMR2_DEPLOYMENT"
