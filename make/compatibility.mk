# Build and test mounted upstream code with the selected Python and CPU PyTorch.
# Capture explicit Make choices as data, following the other operator inputs.
# help: Compatibility PY_VERSION selects a pinned interpreter: 3.12 (default) or 3.13.
# help: Compatibility checks use CHECKPOINT=chefnmr2-epoch949 unless selected explicitly.
# help: Compatibility commands retain the image/build cache and checkpoint volume.
override export CHEFNMR2_COMPATIBILITY_CHECKPOINT := $(if $(filter command line,$(origin CHECKPOINT)),$(value CHECKPOINT),chefnmr2-epoch949)
override export CHEFNMR2_COMPATIBILITY_PY_VERSION_ORIGIN := $(origin PY_VERSION)
override export CHEFNMR2_COMPATIBILITY_PY_VERSION := $(if $(filter command line,$(origin PY_VERSION)),$(value PY_VERSION))
unexport BUILDX_BUILDER BUILDKIT_HOST

# Only the small operation directory is disposable; Docker owns --rm containers.
define compatibility_setup
compatibility_dockerfile=$$CHEFNMR2_REPOSITORY_ROOT/docker/containers/compatibility/Dockerfile
compatibility_version=$$CHEFNMR2_COMPATIBILITY_PY_VERSION
if [[ "$$CHEFNMR2_COMPATIBILITY_PY_VERSION_ORIGIN" != 'command line' ]]; then
    compatibility_version=$$(awk '$$1 == "ARG" && $$2 ~ /^PY_VERSION=/ {sub(/^PY_VERSION=/, "", $$2); print $$2; exit}' "$$compatibility_dockerfile")
fi
compatibility_base=$$(CHEFNMR2_SELECTED_PY_VERSION="$$compatibility_version" awk \
    '$$1 == "FROM" && $$3 == "AS" && $$4 == ("python" ENVIRON["CHEFNMR2_SELECTED_PY_VERSION"]) {print $$2; exit}' "$$compatibility_dockerfile")
if [[ -z "$$compatibility_base" ]]; then
    printf '%s\n' 'Cannot build the compatibility image: PY_VERSION must select a pinned Python version listed in docker/containers/compatibility/Dockerfile.' >&2
    exit 2
fi
$(docker_setup)
compatibility_directory=$$(mktemp -d /tmp/chefnmr2-compatibility.XXXXXXXX)
cleanup_compatibility() {
    operation_status=$$?
    trap - EXIT HUP INT TERM
    cleanup_status=0
    rm -rf -- "$$compatibility_directory" || cleanup_status=$$?
    if (( operation_status != 0 )); then
        exit "$$operation_status"
    fi
    exit "$$cleanup_status"
}
trap cleanup_compatibility EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
compatibility_docker=(docker --host "$$selected_docker_endpoint")
endef

# Resolve the base from the Dockerfile and use the selected engine's builder.
# Run the resulting image by ID so another build cannot replace it under the test.
define compatibility_build
if ! "$${compatibility_docker[@]}" image inspect "$$compatibility_base" >/dev/null; then
    printf '%s\n' 'Cannot build the compatibility image: its pinned Python base must already be available on the selected Docker engine.' >&2
    exit 2
fi
DOCKER_BUILDKIT=1 "$${compatibility_docker[@]}" build --builder default --pull=false --load \
    --build-arg "PY_VERSION=$$compatibility_version" --target check \
    --iidfile "$$compatibility_directory/image.id" --tag "chefnmr2-python$${compatibility_version//./}:check" \
    --file "$$compatibility_dockerfile" \
    "$$CHEFNMR2_REPOSITORY_ROOT/chefnmr2-upstream"
compatibility_image=$$(cat "$$compatibility_directory/image.id")
if [[ ! "$$compatibility_image" =~ ^sha256:[a-f0-9]{64}$$ ]]; then
    printf '%s\n' 'Cannot run the compatibility image: the build did not record an immutable image ID.' >&2
    exit 2
fi
endef

.PHONY: compatibility/image compatibility/check

compatibility/image: ## Build CPU-only PyTorch for the selected PY_VERSION
	@$(compatibility_setup)
	$(compatibility_build)

compatibility/check: ## Build, verify one checkpoint pair, and qualify CPU inference for PY_VERSION
	@$(compatibility_setup)
	checkpoint=$$CHEFNMR2_COMPATIBILITY_CHECKPOINT
	selected_files=0
	while read -r expected_hash maximum_bytes relative_path; do
	    if [[ "$$relative_path" == "$$checkpoint/model.pt" || "$$relative_path" == "$$checkpoint/config.yaml" ]]; then
	        printf '%s  /checkpoints/%s\n' "$$expected_hash" "$$relative_path" >> "$$compatibility_directory/checksums.sha256"
	        ((selected_files += 1))
	    fi
	done < "$$CHEFNMR2_REPOSITORY_ROOT/requirements/checkpoints.sha256"
	if (( selected_files != 2 )); then
	    printf '%s\n' 'Cannot test inference: CHECKPOINT must name a model/config pair in requirements/checkpoints.sha256.' >&2
	    exit 2
	fi
	$(compatibility_build)
	"$${compatibility_docker[@]}" volume inspect chefnmr2-weights_checkpoints >/dev/null
	"$${compatibility_docker[@]}" run --rm --init --pull never --network none --read-only \
	    --log-driver local --log-opt max-size=1m --log-opt max-file=2 --log-opt compress=true \
	    --user 65532:65532 --cap-drop ALL --security-opt no-new-privileges \
	    --pids-limit 64 --cpus 2 --memory 8g --memory-swap 8g \
	    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
	    --mount "type=bind,src=$$CHEFNMR2_REPOSITORY_ROOT/chefnmr2-upstream,dst=/upstream,readonly" \
	    --mount type=volume,src=chefnmr2-weights_checkpoints,dst=/checkpoints,readonly \
	    --mount "type=bind,src=$$compatibility_directory/checksums.sha256,dst=/checksums.sha256,readonly" \
	    --entrypoint /bin/sh "$$compatibility_image" -eu -c \
	    'sha256sum --check /checksums.sha256; exec python -P /usr/local/libexec/check-chefnmr2.py "$$1"' \
	    compatibility-check "/checkpoints/$$checkpoint"
