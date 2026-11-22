# Acquisition consumes a resolved URL; host-specific proxy discovery stays outside it
# Capture the explicit Make choice once rather than inheriting shell proxy settings
# help: Set PROXY_URL=<url> to use a proxy reachable from the download container.
# help: Omit PROXY_URL to be asked; Enter selects none for direct downloads.
# help: Unattended downloads require PROXY_URL=<url> or PROXY_URL=none.
# help: Image pulls use the Docker engine's proxy.
override export CHEFNMR2_PROXY_URL_ORIGIN := $(origin PROXY_URL)
override export PROXY_URL := $(if $(filter command line,$(origin PROXY_URL)),$(value PROXY_URL))

# An explicit empty value is an error; only omission requests a prompt.
define proxy_setup
if [[ "$$CHEFNMR2_PROXY_URL_ORIGIN" != 'command line' ]]; then
    read_operator_input PROXY_URL PROXY_URL 'Download PROXY_URL (URL or none) [none]: '
    PROXY_URL=$${PROXY_URL:-none}
fi
if [[ -z "$$PROXY_URL" ]]; then
    printf '%s\n' 'Cannot download checkpoints: set PROXY_URL=<url> or PROXY_URL=none on the Make command line.' >&2
    exit 2
fi
export PROXY_URL
endef
