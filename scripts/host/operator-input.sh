# Shared host-side terminal input; callers own prompts and value validation.
# Like the related NMR repositories, do not consume redirected recipe stdin.
read_operator_input() {
    local operator_output=$1 operator_name=$2 operator_prompt=$3 operator_terminal operator_answer
    if ! { exec {operator_terminal}<>/dev/tty; } 2>/dev/null; then
        printf 'Cannot ask for %s: set %s on the Make command line when no terminal is available.\n' \
            "$operator_name" "$operator_name" >&2
        return 2
    fi
    printf '%s' "$operator_prompt" >&"$operator_terminal"
    if ! IFS= read -r operator_answer <&"$operator_terminal"; then
        exec {operator_terminal}>&-
        printf 'Cannot read %s: terminal input ended before a value was read.\n' "$operator_name" >&2
        return 2
    fi
    exec {operator_terminal}>&-
    printf -v "$operator_output" '%s' "$operator_answer"
}
