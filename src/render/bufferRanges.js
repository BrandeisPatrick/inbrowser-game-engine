// Allocation-free partial GPU uploads: grow one reused update range per attribute (three clears
// attribute.updateRanges after each upload, so the first touch of a frame starts a fresh range).

/** Mark items [start, start + count) of `attr` (in array elements) dirty, merging into the reused range `r`. */
export function touchRange(attr, r, start, count) {
  const ur = attr.updateRanges;
  if (ur.length !== 1 || ur[0] !== r) {
    ur.length = 0; r.start = start; r.count = count; ur.push(r);
  } else {
    const end = Math.max(r.start + r.count, start + count);
    if (start < r.start) r.start = start;
    r.count = end - r.start;
  }
  attr.needsUpdate = true;
}

/** A reusable { start, count } range object. */
export const newRange = () => ({ start: 0, count: 0 });
