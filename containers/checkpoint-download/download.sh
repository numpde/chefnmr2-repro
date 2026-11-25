#!/bin/sh
# Run in the checkpoint volume; publish only files matching the pinned manifest.
set -eu

download_proxy=${HTTPS_PROXY:?Run make weights/download to acquire checkpoints through the selected route}
checkpoint_release_url=${CHECKPOINT_RELEASE_URL:?Set the pinned checkpoint release URL}
receipt_file=${CHECKPOINT_RECEIPT_FILE:?Set the checkpoint bundle receipt filename}
checksum_file=$1

# Reject links and special files before opening paths in the reused volume.
non_regular_entry=$(find . -xdev ! -type d ! -type f -print -quit)
if test -n "$non_regular_entry"; then
    printf 'Cannot download checkpoints: volume contains a non-regular entry: %s\n' "$non_regular_entry" >&2
    exit 1
fi

chmod 0755 .
while read -r expected_sha256 maximum_file_bytes checkpoint_file; do
    case "$expected_sha256" in ''|'#'*) continue ;; esac
    checkpoint_directory=$(dirname "$checkpoint_file")

    # Directories must be traversable by the model's different non-root user.
    umask 022
    mkdir -p "$checkpoint_directory"
    chmod 0755 "$checkpoint_directory"

    if test -f "$checkpoint_file"; then
        if printf '%s  %s\n' "$expected_sha256" "$checkpoint_file" | sha256sum -c - >/dev/null 2>&1; then
            continue
        fi
        printf 'Restoring pinned source %s\n' "$checkpoint_file"
    fi

    partial_file=$checkpoint_file.part

    # Partial files stay private until their checksum matches.
    umask 077
    # Resume partial files; force the acquisition route and bound each transfer.
    printf 'Downloading %s (resuming any partial file)\n' "$checkpoint_file"
    curl --disable --fail --location --show-error --no-progress-meter \
        --proxy "$download_proxy" --noproxy '' \
        --proto '=https' --proto-redir '=https' \
        --retry 5 --retry-all-errors --retry-delay 2 --retry-max-time 7200 \
        --connect-timeout 15 --max-time 3600 \
        --speed-limit 1024 --speed-time 60 --max-filesize "$maximum_file_bytes" \
        --continue-at - --output "$partial_file" "$checkpoint_release_url/$checkpoint_file"

    printf '%s  %s\n' "$expected_sha256" "$partial_file" | sha256sum -c -
    chmod 0444 "$partial_file"
    # Changing any bundle member invalidates that bundle's receipt.
    rm -f "$checkpoint_directory/$receipt_file"
    sync
    mv "$partial_file" "$checkpoint_file"
done < "$checksum_file"

# sha256sum consumes only the hash and path, not the per-file download cap.
awk 'NF && $1 !~ /^#/ { print $1 "  " $3 }' "$checksum_file" | sha256sum -c -

# Seal only after all files pass verification.
while read -r expected_sha256 maximum_file_bytes checkpoint_file; do
    case "$expected_sha256" in ''|'#'*) continue ;; esac
    checkpoint_directory=$(dirname "$checkpoint_file")
    chmod 0444 "$checkpoint_file"
    chmod 0555 "$checkpoint_directory"
done < "$checksum_file"
chmod 0555 .
