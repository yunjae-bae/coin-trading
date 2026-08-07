#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ $# -gt 0 ]; then
  rel="${1#"$PWD"/}"
  case "$rel" in *.py) ;; *) exit 0 ;; esac
  uv run --group lint ruff check "$1"
else
  uv run --group lint ruff check .
fi

uv run --group lint basedpyright .
