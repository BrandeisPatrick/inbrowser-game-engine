// Clock: game time control (slow motion, pause, frame-by-frame stepping) for the Engine to apply. Pure JS, no
// dependencies, allocation-free. The Clock only does the arithmetic; the Engine wires it into tick():
//
//   dt = clock.advance(raw);    // once per tick: takes the UNSCALED dt, returns the scaled one (0 while paused,
//                               // raw * timeScale otherwise), which feeds the fixed-step accumulator
//   if (clock.paused) run clock.takeSteps() fixed steps (and no accumulator steps)
//   clock.scale(dt) is the same scaling without accumulating anything.
//
//   clock.timeScale = 0.25;     // slow motion; validated on write (NaN, negative or infinite -> 1)
//   clock.pause(); clock.step(); clock.step(5); clock.resume(); clock.toggle();
//
// Fields: timeScale (1), paused (false), time (scaled seconds, summed by advance: does not grow while paused, and
// forced steps do not add to it), unscaledTime (real seconds, summed by advance whether paused or not), frame
// (advance() calls). step(n) queues n forced fixed steps; the Engine takes them only while paused (a step queued
// while running waits for the next paused tick), and resume() drops any still queued (a stale step must not fire on
// a later pause). A dt that is not >= 0 (NaN, negative) counts as 0, so scale() / advance() never return a value
// that could poison the Engine's accumulator.

export class Clock {
  /**
   * @param {object} [o]
   * @param {number} [o.timeScale=1]
   * @param {boolean} [o.paused=false]
   */
  constructor(o = {}) {
    this._timeScale = 1;
    this.timeScale = o.timeScale;
    this.paused = !!o.paused;
    this.time = 0;
    this.unscaledTime = 0;
    this.frame = 0;
    this._steps = 0;
  }

  /** Time multiplier (>= 0, finite; 1 = real time). Invalid writes (NaN, negative, Infinity) set 1. */
  get timeScale() { return this._timeScale; }
  set timeScale(v) { this._timeScale = Number.isFinite(v) && v >= 0 ? v : 1; }

  /** Scaled dt: 0 while paused, else dt * timeScale (dt not >= 0 counts as 0). */
  scale(dt) { return this.paused || !(dt >= 0) ? 0 : dt * this._timeScale; }

  /**
   * Accumulate one tick: takes the UNSCALED dt (seconds; not >= 0 counts as 0) and returns the scaled dt.
   * unscaledTime += dt, time += scaled dt, frame++.
   */
  advance(dt) {
    if (!(dt >= 0)) dt = 0;
    const s = this.paused ? 0 : dt * this._timeScale;
    this.unscaledTime += dt; this.time += s; this.frame++;
    return s;
  }

  /** Queue n (default 1) forced fixed steps; the Engine runs them on the next paused tick. */
  step(n = 1) { n = Math.floor(n); if (n > 0 && n < Infinity) this._steps += n; return this; }

  /** Return the number of queued forced steps and clear the queue. */
  takeSteps() { const n = this._steps; this._steps = 0; return n; }

  /** Queued forced steps (not cleared). */
  get pendingSteps() { return this._steps; }

  pause() { this.paused = true; return this; }
  /** Unpause; also drops queued forced steps. */
  resume() { this.paused = false; this._steps = 0; return this; }
  /** Flip paused; returns the new paused state. */
  toggle() { if (this.paused) this.resume(); else this.pause(); return this.paused; }
}
