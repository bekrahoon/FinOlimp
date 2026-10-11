#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MCP_PROJECT="$ROOT/stage3_ai/integrations/finolimp_mcp"

fail() {
  printf 'ОШИБКА: %s\n' "$*" >&2
  exit 1
}

persist_user_path() {
  local bin_dir="$1"
  local path_line="export PATH=\"$bin_dir:\$PATH\""
  local profile
  for profile in "$HOME/.profile" "$HOME/.bashrc"; do
    if ! grep -Fq -- "$bin_dir" "$profile" 2>/dev/null; then
      printf '\n# FinOlimp local command-line tools\n%s\n' "$path_line" >> "$profile"
    fi
  done
}

[[ -f "$ROOT/requirements-runner.txt" ]] || fail "Скрипт запускайте из копии проекта FinOlimp."
[[ -f "$MCP_PROJECT/pyproject.toml" && -f "$MCP_PROJECT/uv.lock" ]] \
  || fail "Не найдены файлы MCP-проекта: $MCP_PROJECT"

if [[ ! -r /etc/os-release ]] || ! grep -qi '^ID=ubuntu' /etc/os-release; then
  fail "Этот установщик предназначен для Ubuntu."
fi

printf '\n[1/5] Устанавливаю системные утилиты для загрузки пакетов...\n'
sudo apt-get update
sudo apt-get install -y ca-certificates curl unzip

printf '\n[2/5] Устанавливаю uv (менеджер Python и виртуальных окружений)...\n'
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
for bin_dir in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
  if [[ -x "$bin_dir/uv" ]]; then
    persist_user_path "$bin_dir"
  fi
done
command -v uv >/dev/null 2>&1 || fail "uv установлен, но не найден в PATH. Перезапустите терминал и повторите установку."

printf '\n[3/5] Устанавливаю Python 3.12 и зависимости Stage 1/2...\n'
uv python install 3.12
uv venv --python 3.12 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements-runner.txt"

printf '\n[4/5] Создаю отдельное заблокированное окружение MCP...\n'
uv sync --locked --no-dev --project "$MCP_PROJECT" --python 3.12

printf '\n[5/5] Устанавливаю OpenCode CLI...\n'
if ! command -v opencode >/dev/null 2>&1; then
  curl -fsSL https://opencode.ai/v2/install | bash
fi
for bin_dir in "$HOME/.opencode/bin" "$HOME/.local/bin" "$HOME/.local/share/opencode/bin"; do
  if [[ -x "$bin_dir/opencode" ]]; then
    export PATH="$bin_dir:$PATH"
    persist_user_path "$bin_dir"
    break
  fi
done
command -v opencode >/dev/null 2>&1 || fail "OpenCode не найден в PATH после установки. Перезапустите терминал и повторите проверку командой opencode --version."

printf '\nУстановка завершена. Откройте новый терминал, затем из корня проекта:\n'
printf '  .venv/bin/python run_stage_1_2.py\n'
printf '  .venv/bin/python run_ai.py\n\n'
printf 'Перед использованием AI подключите провайдера командой /connect в OpenCode.\n'
