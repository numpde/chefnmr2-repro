# Entry point for commands and help owned by the included Makefiles
override SHELL := /bin/bash
override .SHELLFLAGS := -eu -o pipefail -c
# The exit trap must own the whole recipe, including its Compose command
.ONESHELL:
.DEFAULT_GOAL := help

override export CHEFNMR2_REPOSITORY_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

include $(CHEFNMR2_REPOSITORY_ROOT)/make/docker.mk
include $(CHEFNMR2_REPOSITORY_ROOT)/make/proxy.mk
include $(CHEFNMR2_REPOSITORY_ROOT)/make/weights.mk

shell_quote = '$(subst ','"'"',$(1))'

# Target descriptions (##) and help notes stay beside the code that owns them
.PHONY: help
help:
	@awk -F ':.*## ' '
	    BEGIN { print "Commands:" }
	    /^[[:alnum:]_.\/-]+:.*## / { printf "  make %-20s %s\n", $$1, $$2 }
	    /^# help: / { sub(/^# help: /, ""); notes[++note_count] = $$0 }
	    END {
	        print ""
	        for (i = 1; i <= note_count; i++) print notes[i]
	    }
	' $(foreach makefile,$(MAKEFILE_LIST),$(call shell_quote,$(makefile)))
