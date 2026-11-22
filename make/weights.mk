# One canonical checkout per engine owns the chefnmr2-weights Compose project
# Keep the project name stable so these commands reuse manually downloaded weights
# help: Weight commands retain the checkpoint volume and cached curl image.
define weights_setup
$(docker_setup)
# Cleanup must not stop another weights operation in this checkout
exec 9<"$$CHEFNMR2_REPOSITORY_ROOT"
if ! flock --nonblock 9; then
    printf 'Cannot acquire the weights operation lock for %s.\n' "$$CHEFNMR2_REPOSITORY_ROOT" >&2
    exit 2
fi
weights_compose=(docker --host "$$selected_docker_endpoint" compose --env-file /dev/null \
    --project-name chefnmr2-weights --file "$$CHEFNMR2_REPOSITORY_ROOT/compose/weights.yaml")
endef

# Preserve an operation failure, but fail successful work if cleanup fails
define weights_cleanup_on_exit
cleanup_weights() {
    operation_status=$$?
    trap - EXIT HUP INT TERM
    cleanup_status=0
    "$${weights_compose[@]}" down || cleanup_status=$$?
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

.PHONY: weights/download weights/check weights/config weights/down

weights/download: ## Download and verify checkpoints, then clean up
	@$(weights_setup)
	$(proxy_setup)
	$(weights_cleanup_on_exit)
	"$${weights_compose[@]}" up --abort-on-container-failure verify

weights/check: ## Verify checkpoint hashes without downloading weights, then clean up
	@$(weights_setup)
	$(weights_cleanup_on_exit)
	"$${weights_compose[@]}" run --rm --no-deps verify

weights/config: ## Show the Compose configuration without starting containers
	@$(weights_setup)
	"$${weights_compose[@]}" config

weights/down: ## Remove leftover containers and network
	@$(weights_setup)
	"$${weights_compose[@]}" down
