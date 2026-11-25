#!/usr/bin/env bash
# Checkpoint recipes share the same Compose project, volume and selected engine.

source "$(dirname -- "${BASH_SOURCE[0]}")/operator.sh"

weights_compose() {
    "${docker[@]}" compose --project-directory "$operator_root" --env-file /dev/null \
        -f "$operator_root/compose/weights.yaml" -f "$operator_root/model/weights.yaml" "$@"
}

weights_run() {
    weights_compose run --rm --no-deps -T "$@"
}
