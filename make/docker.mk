# Select a local context and pin its endpoint for the operation and cleanup
# Preserve literal command-line input, then discard ambient Docker/Compose routing
# help: Omit DOCKER_CONTEXT to choose a local context interactively.
# help: Unattended commands require DOCKER_CONTEXT=<local-context>.
override export CHEFNMR2_DOCKER_CONTEXT_INPUT := $(if $(filter command line,$(origin DOCKER_CONTEXT)),$(value DOCKER_CONTEXT))
unexport DOCKER_CONTEXT DOCKER_HOST DOCKER_TLS DOCKER_TLS_VERIFY DOCKER_CERT_PATH
unexport COMPOSE_FILE COMPOSE_PROJECT_NAME COMPOSE_PROFILES COMPOSE_ENV_FILES

# Set selected_docker_context from an explicit menu choice, without a default
define docker_context_menu
# Make may redirect stdin, so use the controlling terminal for both prompt and answer
if ! { exec {menu_terminal}<>/dev/tty; } 2>/dev/null; then
    printf '%s\n' 'Set DOCKER_CONTEXT=<local-context> on the Make command line when no terminal is available.' >&2
    exit 2
fi

configured_contexts=$$(docker context ls --format '{{.Name}} {{.DockerEndpoint}}')
local_context_names=()
while read -r context_name context_endpoint; do
    case "$$context_endpoint" in
        unix:///*) local_context_names+=("$$context_name") ;;
    esac
done <<< "$$configured_contexts"
if (( $${#local_context_names[@]} == 0 )); then
    printf '%s\n' 'Cannot select a Docker context: no local Unix-socket contexts are configured.' >&2
    exit 2
fi

printf '%s\n' 'Select a local Docker context (no default):' >&"$$menu_terminal"
for menu_index in "$${!local_context_names[@]}"; do
    printf '  %d) %s\n' "$$((menu_index + 1))" "$${local_context_names[menu_index]}" >&"$$menu_terminal"
done
printf 'Choice: ' >&"$$menu_terminal"
IFS= read -r menu_choice <&"$$menu_terminal"
exec {menu_terminal}>&-

# Match displayed numbers as text instead of evaluating user input as shell arithmetic
for menu_index in "$${!local_context_names[@]}"; do
    if [[ "$$menu_choice" == "$$((menu_index + 1))" ]]; then
        selected_docker_context=$${local_context_names[menu_index]}
        break
    fi
done
if [[ -z "$$selected_docker_context" ]]; then
    printf 'Cannot select a Docker context: %s must be one of the listed numbers.\n' "$$menu_choice" >&2
    exit 2
fi
endef

# Set selected_docker_endpoint once, before callers issue any engine commands
define docker_setup
selected_docker_context=$$CHEFNMR2_DOCKER_CONTEXT_INPUT
if [[ -z "$$selected_docker_context" ]]; then
    $(docker_context_menu)
fi

selected_docker_endpoint=$$(docker --context "$$selected_docker_context" context inspect \
    "$$selected_docker_context" --format '{{.Endpoints.docker.Host}}')
case "$$selected_docker_endpoint" in
    unix:///*) ;;
    *)
        printf 'Cannot use Docker context %s: endpoint %s must be a local Unix socket.\n' \
            "$$selected_docker_context" "$$selected_docker_endpoint" >&2
        exit 2
        ;;
esac
endef
