SHELL := /bin/bash
.DEFAULT_GOAL := help
export CHEFNMR2_PROFILE := $(value PROFILE)
export CHEFNMR2_CONTEXT := $(value DOCKER_CONTEXT)
export CHEFNMR2_POOL := $(value WHEELHOUSE)

.PHONY: help sdk/acquire lock acquire build check
help:
	@echo 'make sdk/acquire DOCKER_CONTEXT=science WHEELHOUSE=/cache/pool'
	@echo 'make lock|acquire|build PROFILE=provider|worker-cpu DOCKER_CONTEXT=science WHEELHOUSE=/cache/pool'
	@echo 'make check DOCKER_CONTEXT=science'

sdk/acquire lock acquire build:
	@python3 -B scripts/build.py $@ --profile "$$CHEFNMR2_PROFILE" --context "$$CHEFNMR2_CONTEXT" --pool "$$CHEFNMR2_POOL" $(if $(filter lock,$@),--offline,)

check:
	@python3 -B scripts/runtime.py check --context "$$CHEFNMR2_CONTEXT"
