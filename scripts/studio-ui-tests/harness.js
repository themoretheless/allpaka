// Shared helpers for the Studio UI tests. Each test loads slices of the web
// sources into a node:vm context; these helpers read the sources once and
// fail loudly when a marker the slice depends on has moved or been renamed,
// instead of silently evaluating the wrong part of app.js.
'use strict';
const fs = require('fs');

const cache = new Map();

/** Source of a file under crates/allpaka-chat/web, read once per process. */
function webSource(name) {
  if (!cache.has(name)) {
    cache.set(name, fs.readFileSync(`crates/allpaka-chat/web/${name}`, 'utf8'));
  }
  return cache.get(name);
}

/** crates/allpaka-chat/web/app.js. */
function appSource() {
  return webSource('app.js');
}

/**
 * `source.slice(source.indexOf(start), source.indexOf(end))`, but both
 * markers must exist and `end` must come after `start`.
 */
function between(source, start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end);
  if (from < 0) throw new Error(`UI test marker not found: ${JSON.stringify(start)}`);
  if (to < 0) throw new Error(`UI test marker not found: ${JSON.stringify(end)}`);
  if (to <= from) {
    throw new Error(`UI test marker ${JSON.stringify(end)} precedes ${JSON.stringify(start)}`);
  }
  return source.slice(from, to);
}

/** Let pending promise callbacks and immediates run. */
function flush() {
  return new Promise(resolve => setImmediate(resolve));
}

module.exports = { webSource, appSource, between, flush };
