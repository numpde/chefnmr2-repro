# Build and test the mounted upstream on Python 3.12 with CPU-only PyTorch.
# Capture an explicit checkpoint choice as data, following the other Make inputs.
# help: Compatibility checks use CHECKPOINT=chefnmr2-epoch949 unless selected explicitly.
# help: Compatibility commands retain the image/build cache and checkpoint volume.
override export CHEFNMR2_COMPATIBILITY_CHECKPOINT := $(if $(filter command line,$(origin CHECKPOINT)),$(value CHECKPOINT),chefnmr2-epoch949)
unexport BUILDX_BUILDER BUILDKIT_HOST

# Only the small operation directory is disposable; Docker owns --rm containers.
define compatibility_setup
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
compatibility_base=$$(awk '$$1 == "FROM" {print $$2; exit}' "$$CHEFNMR2_REPOSITORY_ROOT/containers/compatibility/Dockerfile")
if ! "$${compatibility_docker[@]}" image inspect "$$compatibility_base" >/dev/null; then
    printf '%s\n' 'Cannot build the compatibility image: its pinned Python base must already be available on the selected Docker engine.' >&2
    exit 2
fi
DOCKER_BUILDKIT=1 "$${compatibility_docker[@]}" build --builder default --pull=false --load \
    --iidfile "$$compatibility_directory/image.id" --tag chefnmr2-python312:check \
    --file "$$CHEFNMR2_REPOSITORY_ROOT/containers/compatibility/Dockerfile" \
    "$$CHEFNMR2_REPOSITORY_ROOT/chefnmr2-upstream"
compatibility_image=$$(cat "$$compatibility_directory/image.id")
if [[ ! "$$compatibility_image" =~ ^sha256:[a-f0-9]{64}$$ ]]; then
    printf '%s\n' 'Cannot run the compatibility image: the build did not record an immutable image ID.' >&2
    exit 2
fi
endef

.PHONY: compatibility/image compatibility/check

compatibility/image: ## Build the Python 3.12 CPU-only PyTorch compatibility image
	@$(compatibility_setup)
	$(compatibility_build)

compatibility/check: ## Build, verify one checkpoint pair, and run CPU inference on Python 3.12
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
	    --user 65532:65532 --cap-drop ALL --security-opt no-new-privileges \
	    --pids-limit 64 --cpus 2 --memory 8g --memory-swap 8g \
	    --tmpfs /tmp:rw,nosuid,nodev,noexec,size=64m \
	    --mount "type=bind,src=$$CHEFNMR2_REPOSITORY_ROOT/chefnmr2-upstream,dst=/upstream,readonly" \
	    --mount type=volume,src=chefnmr2-weights_checkpoints,dst=/checkpoints,readonly \
	    --mount "type=bind,src=$$compatibility_directory/checksums.sha256,dst=/checksums.sha256,readonly" \
	    --entrypoint /bin/sh "$$compatibility_image" -eu -c \
	    'sha256sum --check /checksums.sha256; exec python -P /usr/local/libexec/check-chefnmr2.py "$$1"' \
	    compatibility-check "/checkpoints/$$checkpoint"
