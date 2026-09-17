#!/usr/bin/env bash
# Free RAM on the SpacemiT K3 by stopping the bundled AI-assistant / desktop service stack.
# Everything here is reversible: scripts/restore-services.sh puts it all back.
set -u

UNITS=(
  # Bianbu Agent stack (memory adapter + embedding llama-server + runtime)
  bianbu-agent-memory-adapter.service
  bianbu-agent-memory-embedding.service
  bianbu-agent-memoryd.service
  bianbu-agent-network-learning.service
  bianbu-agent-runtime@bianbu.service
  # BiBit assistant (audio 840MB, GUI, file search, intent)
  bibit-runtime-adapter@bianbu.service
  # Local document->markdown converter (1.6GB RSS)
  file2md.service
  # Local metasearch engine
  searxng.service
)

echo "=== before ==="
free -h | head -2

for u in "${UNITS[@]}"; do
  if systemctl list-unit-files "$u" >/dev/null 2>&1; then
    sudo systemctl stop "$u" 2>/dev/null
    sudo systemctl disable "$u" 2>/dev/null
    printf 'stopped+disabled %s\n' "$u"
  fi
done

# User-session assistant processes that are not covered by the units above.
for pat in 'bibit-' 'bianbu-agent' '/opt/file2md/' 'searxng'; do
  pkill -f "$pat" 2>/dev/null && printf 'killed pattern %s\n' "$pat"
done

sleep 2
echo "=== after ==="
free -h | head -2
