override SHELL := /bin/bash
override .SHELLFLAGS := -eu -o pipefail -c
override .DEFAULT_GOAL := help
override REPOSITORY_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
include $(REPOSITORY_ROOT)/project.mk
override HOST_PYTHON := python3 -B
override DEPLOYMENTS := $(notdir $(patsubst %/,%,$(wildcard $(REPOSITORY_ROOT)/deployments/*/)))
override PROFILES := provider worker-cpu worker-cuda weights-sanitize

# Capture literal command-line inputs before Make or the shell can interpret them.
# Omitted inputs are selected at the terminal by the function below.
override export REPRO_EXPLICIT_INPUTS := $(foreach name,DOCKER_CONTEXT PROXY_URL WHEELHOUSE PROFILE DEPLOYMENT,$(if $(filter command line,$(origin $(name))),$(name)))
override export REPRO_DOCKER_CONTEXT := $(value DOCKER_CONTEXT)
override export REPRO_PROXY_URL := $(value PROXY_URL)
override export REPRO_WHEELHOUSE := $(value WHEELHOUSE)
override export REPRO_PROFILE := $(value PROFILE)
override export REPRO_DEPLOYMENT := $(value DEPLOYMENT)
ifeq ($(origin ASSUME_GPU),command line)
override export REPRO_ASSUME_GPU := $(value ASSUME_GPU)
else
override export REPRO_ASSUME_GPU := 0
endif
unexport DOCKER_CONTEXT PROXY_URL WHEELHOUSE PROFILE DEPLOYMENT ASSUME_GPU

.PHONY: help
help: ## List workflows; inputs may prompt on a terminal and environment values never select them.
	@awk -F ':.*## ' '/^[a-z][a-z0-9_./-]*:.*## / { printf "  make %-28s %s\n", $$1, $$2 }' "$(REPOSITORY_ROOT)/Makefile"

.PHONY: upstream/init
upstream/init: ## Initialize the pinned upstream submodule with a shallow checkout; may use the network.
	git -C "$(REPOSITORY_ROOT)" submodule sync -- model/upstream
	git -C "$(REPOSITORY_ROOT)" submodule update --init --checkout --depth 1 -- model/upstream

# Shared shell behavior stays in shell files; Make only passes literal inputs.
override operator_inputs := source "$(REPOSITORY_ROOT)/make/operator.sh"
override weights_inputs := source "$(REPOSITORY_ROOT)/make/weights.sh"

.PHONY: images/pull
images/pull: ## Pull pinned Python bases and checkpoint downloader; requires DOCKER_CONTEXT.
	@$(operator_inputs); select_docker; \
	"$${docker[@]}" compose --project-directory "$(REPOSITORY_ROOT)" --env-file /dev/null \
		-f "$(REPOSITORY_ROOT)/compose/images.yaml" pull --ignore-buildable

.PHONY: gpu/check
gpu/check: ## Require a host NVIDIA GPU; ASSUME_GPU=1 explicitly skips the probe (default 0).
	@$(operator_inputs); check_gpu "$${REPRO_ASSUME_GPU}"

.PHONY: wheelhouse/acquire prepare lock acquire build
wheelhouse/acquire: ## Verify or acquire independent Wheelhouse tooling; requires DOCKER_CONTEXT.
	@$(operator_inputs); pool=$$(choose WHEELHOUSE); proxy=$$(choose PROXY_URL); select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/build.py" wheelhouse/acquire --host "$$docker_host" --pool "$$pool" --proxy "$$proxy"

.PHONY: antlr/acquire antlr/prepare
antlr/acquire: ## Acquire pinned Wheelhouse, ANTLR source and locked build dependencies.
	@$(operator_inputs); pool=$$(choose WHEELHOUSE); proxy=$$(choose PROXY_URL); select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/build.py" antlr/acquire --host "$$docker_host" --pool "$$pool" --proxy "$$proxy"

antlr/prepare: ## Build ANTLR offline and retain a private wheel, metadata and receipt.
	@$(operator_inputs); pool=$$(choose WHEELHOUSE); select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/build.py" antlr/prepare --host "$$docker_host" --pool "$$pool"

prepare: ## Prepare exact declared wheel inputs before metadata-only locking; requires PROFILE.
lock: ## Resolve a complete wheel-only profile lock from declared sources; requires PROFILE, WHEELHOUSE and DOCKER_CONTEXT.
acquire: ## Acquire the profile's locked packages into WHEELHOUSE; requires PROFILE and DOCKER_CONTEXT.
build: ## Build offline; worker-cuda requires gpu/check unless ASSUME_GPU=1 (default 0).
prepare lock acquire build:
	@$(operator_inputs); profile=$$(choose PROFILE '$(PROFILES)'); \
	if [[ $@ == build && $$profile == worker-cuda ]]; then \
	    env -u MAKEFLAGS -u MFLAGS -u GNUMAKEFLAGS -u MAKEOVERRIDES -u MAKEFILES -u MAKELEVEL \
	        make --no-print-directory -f "$(REPOSITORY_ROOT)/Makefile" gpu/check "ASSUME_GPU=$${REPRO_ASSUME_GPU}"; \
	fi; \
	pool=$$(choose WHEELHOUSE); proxy_args=(); \
	if [[ $@ != build ]]; then proxy_args=(--proxy "$$(choose PROXY_URL)"); fi; select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/build.py" "$@" --profile "$$profile" --host "$$docker_host" --pool "$$pool" "$${proxy_args[@]}"

.PHONY: check test/provider test/provider/normative test/provider/adversarial
check: test/provider ## Run all provider image tests; requires DOCKER_CONTEXT.
test/provider: ## Run normative and adversarial provider image tests; requires DOCKER_CONTEXT.
test/provider/normative: ## Run normative provider image tests; requires DOCKER_CONTEXT.
test/provider/adversarial: ## Run adversarial provider image tests; requires DOCKER_CONTEXT.
test/provider test/provider/normative test/provider/adversarial:
	@$(operator_inputs); select_docker; category='$(@F)'; \
	if [[ $$category == provider ]]; then category=all; fi; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/runtime.py" check --host "$$docker_host" --category "$$category"

.PHONY: test/operator test/operator/normative test/operator/adversarial
test/operator: test/operator/normative test/operator/adversarial ## Run normative and adversarial host operator tests.
test/operator/normative: ## Run normative host operator tests.
test/operator/adversarial: ## Run adversarial host operator tests.
test/operator/normative test/operator/adversarial:
	@PYTHONPATH="$(REPOSITORY_ROOT)/tests/operator:$(REPOSITORY_ROOT)/make" $(HOST_PYTHON) -m unittest discover -s "$(REPOSITORY_ROOT)/tests/operator/$(@F)" -p 'test_*.py' -v

.PHONY: test/model up config manual/config status logs manual
test/model: ## Run real inference and SDK batch tests; requires DEPLOYMENT and DOCKER_CONTEXT.
up: ## Start built services after clean provider retirement; requires DEPLOYMENT and DOCKER_CONTEXT.
config: ## Render the API Compose configuration; requires DEPLOYMENT and DOCKER_CONTEXT.
manual/config: ## Render the manual Compose configuration; requires DEPLOYMENT and DOCKER_CONTEXT.
status: ## Show API service state; requires DEPLOYMENT and DOCKER_CONTEXT.
logs: ## Follow bounded provider and worker logs; requires DEPLOYMENT and DOCKER_CONTEXT.
manual: ## Run the selected provider image as a manual model client; requires DEPLOYMENT and DOCKER_CONTEXT.
test/model up config manual/config status logs manual:
	@$(operator_inputs); deployment=$$(choose DEPLOYMENT '$(DEPLOYMENTS)'); select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/runtime.py" "$@" --host "$$docker_host" --deployment "$$deployment"

.PHONY: weights/download weights/verify weights/sanitize
weights/download: ## Acquire and verify checkpoints; requires PROXY_URL and DOCKER_CONTEXT.
	@$(weights_inputs); proxy=$$(choose PROXY_URL); select_docker; \
	weights_run --pull missing volume-init; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/weights.py" --host "$$docker_host" --proxy "$$proxy"

weights/verify: ## Verify the retained checkpoint volume offline; requires DOCKER_CONTEXT.
weights/sanitize: ## Sanitize known checkpoint files in the retained volume; requires DOCKER_CONTEXT.
weights/verify weights/sanitize:
	@$(weights_inputs); select_docker; \
	printf 'Starting %s\n' '$@' >&2; \
	if [[ $@ == weights/verify ]]; then weights_run --pull never verify; else weights_run --pull never --name "$${IMAGE_NAMESPACE}-weights-writer" sanitize; fi; \
	printf '%s complete\n' '$@' >&2

.PHONY: weights/watch
weights/watch: ## Watch checkpoint file sizes during a download; requires DOCKER_CONTEXT. Ctrl+C stops watching.
	@$(operator_inputs); select_docker; \
	writer=$$("$${docker[@]}" ps --filter "name=^/$${IMAGE_NAMESPACE}-weights-writer$$" --format '{{.ID}}'); \
	if [[ -z $$writer ]]; then printf '%s\n' 'Cannot watch checkpoints: no weights download is running on this engine.' >&2; exit 2; fi; \
	watch --interval 2 --exec "$${docker[@]}" exec "$$writer" find /checkpoints -type f -exec ls -lh '{}' +

.PHONY: test/unit test/unit/normative test/unit/adversarial
test/unit: test/unit/normative test/unit/adversarial ## Run binding tests in the built CPU worker image.
test/unit/normative: ## Run normative binding tests in the built CPU worker; requires DOCKER_CONTEXT.
test/unit/adversarial: ## Run adversarial binding tests in the built CPU worker; requires DOCKER_CONTEXT.
test/unit/normative test/unit/adversarial:
	@$(operator_inputs); select_docker; \
	compose=("$${docker[@]}" compose --project-directory "$(REPOSITORY_ROOT)" --env-file /dev/null -f "$(REPOSITORY_ROOT)/compose/test-unit.yaml"); \
	"$${compose[@]}" run --rm --no-deps --pull never --interactive=false -T binding \
		-B -m unittest discover -s "/tests/unit/$(@F)" -p 'test_binding.py' -v
