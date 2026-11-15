#!/usr/bin/env bash
# Fault injection: hard-kill one node mid-run, then require a successful pipeline anyway.
# usage: scripts/fault_test.sh <service> <process-pattern> <log line to wait for before killing>
# e.g.   scripts/fault_test.sh consumer lib/consumer/raceline_node "centerline: 500 pts"
set -u
cd "$(dirname "$0")/.."
svc=$1 pattern=$2 trigger=$3
docker compose down -t 1 >/dev/null 2>&1
rm -f output/results.json 2>/dev/null
docker compose up -d >/dev/null 2>&1
for _ in $(seq 60); do docker compose logs 2>/dev/null | grep -q "$trigger" && break; sleep 0.5; done
docker compose exec -T "$svc" pkill -9 -f "$pattern" && echo "killed $pattern in $svc"
for _ in $(seq 120); do [ "$(docker inspect -f '{{.State.Status}}' bgr-race-consumer-1)" = exited ] && break; sleep 1; done
sleep 2
docker compose logs 2>/dev/null | grep -E "restart|failed|status|deadline|Restarting" | cut -c1-160
echo "consumer exit=$(docker inspect -f '{{.State.ExitCode}}' bgr-race-consumer-1)" \
     "producer exit=$(docker inspect -f '{{.State.ExitCode}}' bgr-race-producer-1)" \
     "producer restarts=$(docker inspect -f '{{.RestartCount}}' bgr-race-producer-1)" \
     "results.json=$([ -f output/results.json ] && echo yes || echo NO)"
docker compose down -t 1 >/dev/null 2>&1
