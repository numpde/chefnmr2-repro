# One canonical checkout per engine owns the chefnmr2-weights Compose project
# Keep the project name stable so these commands reuse manually downloaded weights
# help: Weight commands retain the checkpoint volume and cached curl image.
# Keep checkpoint readers and writers from racing in the canonical checkout.
define weights_lock
exec 9<"$$CHEFNMR2_REPOSITORY_ROOT"
if ! flock --nonblock 9; then
    printf 'Cannot acquire the weights operation lock for %s.\n' "$$CHEFNMR2_REPOSITORY_ROOT" >&2
    exit 2
fi
endef

define weights_setup
$(docker_setup)
$(weights_lock)
weights_compose=(docker --host "$$selected_docker_endpoint" compose --env-file /dev/null \
    --project-name chefnmr2-weights --file "$$CHEFNMR2_REPOSITORY_ROOT/compose/weights.yaml")
endef

# Preserve an operation failure, but fail successful work if cleanup fails
define weights_cleanup_on_exit
cleanup_weights() {
    operation_status=$$?
    trap - EXIT HUP INT TERM
    cleanup_status=0
    "$${weights_compose[@]}" down --remove-orphans || cleanup_status=$$?
    if (( operation_status != 0 )); then
        exit "$$operation_status"
    fi
    exit "$$cleanup_status"
}
trap cleanup_weights EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
endef

.PHONY: weights/download weights/check weights/config weights/down weights/watch

weights/download: ## Download and verify checkpoints, then clean up
	@$(weights_setup)
	$(proxy_setup)
	$(weights_cleanup_on_exit)
	"$${weights_compose[@]}" run --rm --no-deps volume-init
	"$${weights_compose[@]}" run --rm --no-deps download
	"$${weights_compose[@]}" run --rm --no-deps verify

weights/check: ## Verify checkpoint hashes without downloading weights, then clean up
	@$(weights_setup)
	$(weights_cleanup_on_exit)
	"$${weights_compose[@]}" run --rm --no-deps verify

weights/config: ## Show the Compose configuration without starting containers
	@$(weights_setup)
	"$${weights_compose[@]}" config

weights/down: ## Remove leftover containers and network
	@$(weights_setup)
	"$${weights_compose[@]}" down --remove-orphans

# Observe the selected engine without acquiring the active operation's lock.
weights/watch: ## Watch checkpoint file sizes during a download; Ctrl+C stops watching
	@$(docker_setup)
	download_id=$$(docker --host "$$selected_docker_endpoint" ps \
	    --filter label=com.docker.compose.project=chefnmr2-weights \
	    --filter label=com.docker.compose.service=download --format '{{.ID}}')
	if [[ -z "$$download_id" || "$$download_id" == *$$'\n'* ]]; then
	    printf '%s\n' 'Cannot watch checkpoints: the selected Docker engine must have exactly one running download.' >&2
	    exit 2
	fi
	watch --interval 2 --exec docker --host "$$selected_docker_endpoint" exec "$$download_id" \
	    find /checkpoints -type f -exec ls -lh '{}' +
