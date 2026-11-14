// The streaming answer arrives as newline-delimited JSON: one event per line. Network chunks don't line up
// with lines, so keep the unfinished tail and read it again once more bytes arrive.

export function drainNdjson(buffer: string): { events: unknown[]; rest: string } {
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  const events: unknown[] = [];
  for (const line of lines) {
    const text = line.trim();
    if (!text) continue;
    try {
      events.push(JSON.parse(text));
    } catch {
      /* skip a damaged line rather than lose the rest of the answer */
    }
  }
  return { events, rest };
}
