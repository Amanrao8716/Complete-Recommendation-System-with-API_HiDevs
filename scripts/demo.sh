#!/usr/bin/env bash
# Scripted walkthrough for the demo video.  Start the server first:
#   python scripts/seed_data.py --reset && python -m api.app
BASE="${BASE_URL:-http://127.0.0.1:5000}"
KEY_HEADER=()
[ -n "$API_KEY" ] && KEY_HEADER=(-H "X-API-Key: $API_KEY")
run() { echo; echo "\$ $*"; "$@"; echo; }

run curl -s "$BASE/health"
echo "--- 1. Recommendations for an existing user"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/1?limit=3"
echo "--- 2. Same request again: served from cache"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/1?limit=3"
echo "--- 3. Cold-start user (no history)"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/41?limit=3"
echo "--- 4. Feedback, then recommendations change"
run curl -s -X POST "${KEY_HEADER[@]}" -H "Content-Type: application/json" \
  -d '{"user_id": 41, "content_id": 7, "type": "complete"}' "$BASE/feedback"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/41?limit=3"
echo "--- 5. Error handling"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/9999"
run curl -s "${KEY_HEADER[@]}" "$BASE/recommend/1?limit=0"
echo "--- 6. Metrics"
run curl -s "${KEY_HEADER[@]}" "$BASE/metrics"
