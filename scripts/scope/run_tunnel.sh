#!/usr/bin/env bash
# Keep the Mac-to-Linux reverse tunnel alive inside its own tmux session.
set -euo pipefail

scope_ssh_config=""
scope_host=bupt
scope_child_pid=""
scope_stop_requested=0
scope_attempt=0
scope_backoff=2

scope_usage() {
  printf 'Usage: run_tunnel.sh --ssh-config PATH [--host SSH_ALIAS]\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-config)
      [[ $# -ge 2 ]] || { scope_usage >&2; exit 2; }
      scope_ssh_config=$2
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || { scope_usage >&2; exit 2; }
      scope_host=$2
      shift 2
      ;;
    --help|-h)
      scope_usage
      exit 0
      ;;
    *)
      scope_usage >&2
      exit 2
      ;;
  esac
done

[[ -f "$scope_ssh_config" && -r "$scope_ssh_config" ]] || {
  printf 'A readable SSH config file is required.\n' >&2
  exit 2
}
[[ "$scope_host" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]] || {
  printf 'A valid SSH host alias is required.\n' >&2
  exit 2
}
command -v ssh >/dev/null || {
  printf 'The ssh executable is required.\n' >&2
  exit 2
}

scope_log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1"
}

scope_request_stop() {
  scope_stop_requested=1
  if [[ -n "$scope_child_pid" ]]; then
    # This PID belongs to the ssh or backoff sleep started by this helper.
    kill -TERM "$scope_child_pid" 2>/dev/null || true
  fi
}

scope_wait_for_child() {
  local scope_child_exit=0
  if wait "$scope_child_pid"; then
    scope_child_exit=0
  else
    scope_child_exit=$?
  fi
  if [[ "$scope_stop_requested" -eq 1 ]]; then
    # A signal can interrupt wait before the owned child has exited; reap it.
    wait "$scope_child_pid" 2>/dev/null || true
  fi
  scope_child_pid=""
  return "$scope_child_exit"
}

trap scope_request_stop INT TERM EXIT

while [[ "$scope_stop_requested" -eq 0 ]]; do
  scope_attempt=$((scope_attempt + 1))
  scope_log "attempt=$scope_attempt state=starting"
  # Forwarding and keepalive settings come from the existing private config.
  # Suppress ssh output: only sanitized timestamps and exit codes are logged.
  ssh -N -T -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -F "$scope_ssh_config" "$scope_host" >/dev/null 2>&1 &
  scope_child_pid=$!
  [[ "$scope_stop_requested" -eq 0 ]] || scope_request_stop
  if scope_wait_for_child; then
    scope_ssh_exit=0
  else
    scope_ssh_exit=$?
  fi
  scope_log "attempt=$scope_attempt exit_code=$scope_ssh_exit"
  [[ "$scope_stop_requested" -eq 0 ]] || break

  # A successful disconnect still requires reconnection, with a fresh delay.
  [[ "$scope_ssh_exit" -ne 0 ]] || scope_backoff=2
  scope_log "attempt=$scope_attempt state=backoff seconds=$scope_backoff"
  sleep "$scope_backoff" >/dev/null 2>&1 &
  scope_child_pid=$!
  [[ "$scope_stop_requested" -eq 0 ]] || scope_request_stop
  if scope_wait_for_child; then
    scope_sleep_exit=0
  else
    scope_sleep_exit=$?
  fi
  [[ "$scope_stop_requested" -eq 0 ]] || break
  if [[ "$scope_sleep_exit" -ne 0 ]]; then
    scope_log "attempt=$scope_attempt state=backoff_failed exit_code=$scope_sleep_exit"
    exit 1
  fi
  if [[ "$scope_ssh_exit" -ne 0 ]]; then
    scope_backoff=$((scope_backoff * 2))
    [[ "$scope_backoff" -le 30 ]] || scope_backoff=30
  fi
done

scope_log 'state=stopped'
