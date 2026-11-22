# Acquisition consumes a resolved URL; host-specific proxy discovery stays outside it
# Capture the explicit Make choice once rather than inheriting shell proxy settings
# help: Set PROXY_URL=<url> to use a proxy reachable from the download container.
# help: Omit PROXY_URL for direct downloads. Image pulls use the Docker engine's proxy.
override export PROXY_URL := $(if $(filter command line,$(origin PROXY_URL)),$(value PROXY_URL))
