override SHELL := /bin/bash
override .SHELLFLAGS := -eu -o pipefail -c
override .DEFAULT_GOAL := help
override REPOSITORY_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
override HOST_PYTHON := python3 -B
override DEPLOYMENTS := $(notdir $(patsubst %/,%,$(wildcard $(REPOSITORY_ROOT)/deployments/*/)))
override PROFILES := $(notdir $(patsubst %/,%,$(wildcard $(REPOSITORY_ROOT)/builds/*/)))

# Capture literal command-line inputs before Make or the shell can interpret them.
# Omitted inputs are selected at the terminal by the function below.
override export REPRO_EXPLICIT_INPUTS := $(foreach name,DOCKER_CONTEXT PROXY_URL WHEELHOUSE PROFILE DEPLOYMENT,$(if $(filter command line,$(origin $(name))),$(name)))
override export REPRO_DOCKER_CONTEXT := $(value DOCKER_CONTEXT)
override export REPRO_PROXY_URL := $(value PROXY_URL)
override export REPRO_WHEELHOUSE := $(value WHEELHOUSE)
override export REPRO_PROFILE := $(value PROFILE)
override export REPRO_DEPLOYMENT := $(value DEPLOYMENT)
unexport DOCKER_CONTEXT PROXY_URL WHEELHOUSE PROFILE DEPLOYMENT

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

.PHONY: sdk/acquire lock acquire build
sdk/acquire: ## Verify or acquire pinned SDK tooling in WHEELHOUSE; requires DOCKER_CONTEXT.
	@$(operator_inputs); pool=$$(choose WHEELHOUSE); proxy=$$(choose PROXY_URL); select_docker; \
	$(HOST_PYTHON) "$(REPOSITORY_ROOT)/make/build.py" sdk/acquire --host "$$docker_host" --pool "$$pool" --proxy "$$proxy"

lock: ## Resolve a profile lock from cached packages; requires PROFILE, WHEELHOUSE and DOCKER_CONTEXT.
acquire: ## Acquire the profile's locked packages into WHEELHOUSE; requires PROFILE and DOCKER_CONTEXT.
build: ## Build a selected profile offline; requires PROFILE, WHEELHOUSE and DOCKER_CONTEXT.
lock acquire build:
	@$(operator_inputs); profile=$$(choose PROFILE '$(PROFILES)'); pool=$$(choose WHEELHOUSE); proxy_args=(); \
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

.PHONY: test/model config manual/config status logs manual
test/model: ## Run real inference and SDK batch tests; requires DEPLOYMENT and DOCKER_CONTEXT.
config: ## Render the API Compose configuration; requires DEPLOYMENT and DOCKER_CONTEXT.
manual/config: ## Render the manual Compose configuration; requires DEPLOYMENT and DOCKER_CONTEXT.
status: ## Show API service state; requires DEPLOYMENT and DOCKER_CONTEXT.
logs: ## Follow bounded provider and worker logs; requires DEPLOYMENT and DOCKER_CONTEXT.
manual: ## Run the selected provider image as a manual model client; requires DEPLOYMENT and DOCKER_CONTEXT.
test/model config manual/config status logs manual:
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
	if [[ $@ == weights/verify ]]; then weights_run --pull never verify; else weights_run --pull never --name chefnmr2-weights-writer sanitize; fi; \
	printf '%s complete\n' '$@' >&2

.PHONY: weights/watch
weights/watch: ## Watch checkpoint file sizes during a download; requires DOCKER_CONTEXT. Ctrl+C stops watching.
	@$(operator_inputs); select_docker; \
	writer=$$("$${docker[@]}" ps --filter 'name=^/chefnmr2-weights-writer$$' --format '{{.ID}}'); \
	if [[ -z $$writer ]]; then printf '%s\n' 'Cannot watch checkpoints: no weights download is running on this engine.' >&2; exit 2; fi; \
	watch --interval 2 --exec "$${docker[@]}" exec "$$writer" find /checkpoints -type f -exec ls -lh '{}' +

.PHONY: test/unit test/unit/normative test/unit/adversarial
test/unit: test/unit/normative test/unit/adversarial ## Run sanitizer and worker admission tests in their own images.
test/unit/normative: ## Run normative checkpoint tests in the sanitizer and worker; requires DOCKER_CONTEXT.
test/unit/adversarial: ## Run adversarial checkpoint tests in the sanitizer and worker; requires DOCKER_CONTEXT.
test/unit/normative test/unit/adversarial:
	@$(operator_inputs); select_docker; \
	"$${docker[@]}" run --rm --pull=never --network none --read-only \
		--cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 1 --memory 1g \
		--tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m -e PYTHONPATH=/tests/unit:/opt/sanitize \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/tests/unit,dst=/tests/unit,readonly" \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/containers/weights-sanitize/sanitize.py,dst=/opt/sanitize/sanitize.py,readonly" \
		--entrypoint python chefnmr2/weights-sanitizer:local \
		-B -m unittest discover -s "/tests/unit/$(@F)" -p 'test_checkpoints.py' -v; \
	"$${docker[@]}" run --rm --pull=never --network none --read-only \
		--cap-drop ALL --security-opt no-new-privileges --pids-limit 64 --cpus 1 --memory 1g \
		--tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m -e PROVIDER_CORE_IN_CONTAINER=1 \
		-e PYTHONPATH=/tests/unit:/opt/inference:/opt/sanitize \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/tests/unit,dst=/tests/unit,readonly" \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/model/binding.py,dst=/opt/inference/binding.py,readonly" \
		--mount "type=bind,src=$(REPOSITORY_ROOT)/containers/weights-sanitize/sanitize.py,dst=/opt/sanitize/sanitize.py,readonly" \
		--entrypoint python chefnmr2/worker-cpu:local \
		-B -m unittest discover -s "/tests/unit/$(@F)" -p 'test_binding.py' -v
