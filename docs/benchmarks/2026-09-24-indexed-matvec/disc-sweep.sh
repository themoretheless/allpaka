#!/bin/zsh
# Parallelism-vs-arithmetic discriminator on the streamed indexed shapes.
# Every ALLPAKA_* knob is a OnceLock read, so one config is one process. Each
# process internally interleaves the five formats 4x and reports the best, so
# desktop drift is common-mode; running the whole order twice checks that.
cd /Users/themoretheless/Documents/Sources/allpaka || exit 1
run() {
  print "### $1 :: ${2:-defaults}"
  if [[ -z "$2" ]]; then
    cargo test -q -p allpaka-backend --test gpu_ffnbench -- \
      --ignored --nocapture --test-threads=1 indexed_matvecs 2>&1 | grep -E 'GB/s best|no Metal'
  else
    env ${=2} cargo test -q -p allpaka-backend --test gpu_ffnbench -- \
      --ignored --nocapture --test-threads=1 indexed_matvecs 2>&1 | grep -E 'GB/s best|no Metal'
  fi
}
for pass in A B; do
  print "===== pass $pass ====="
  run default
  run q2nr1 'ALLPAKA_Q2_NR0=1'
  run q2nr4 'ALLPAKA_Q2_NR0=4'
  run q4nr1 'ALLPAKA_Q4_NR0=1'
  run q4nr4 'ALLPAKA_Q4_NR0=4'
  run q5nr1 'ALLPAKA_Q5_NR0=1'
  run q5nr4 'ALLPAKA_Q5_NR0=4'
  run q8nr1 'ALLPAKA_Q8_NR0=1'
  run q8nr4 'ALLPAKA_Q8_NR0=4'
  run lprdiv2 'ALLPAKA_LPR_DIV=2'
  run tg64 'ALLPAKA_MV_TG=64'
  run tg256 'ALLPAKA_MV_TG=256'
  run flatid 'ALLPAKA_MV_ID=0'
done
