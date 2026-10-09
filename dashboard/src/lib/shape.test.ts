import { describe, expect, it } from "vitest";
import { fmt, niceTicks, pct, ratioRange, series, toRows } from "./shape";

describe("series and toRows", () => {
  it("pairs values with x in order and keys rows by x", () => {
    const s = series([8, 16], { random: [1, 2], prefix: [3, 4] });
    expect(s).toEqual([
      { name: "random", points: [{ x: 8, value: 1 }, { x: 16, value: 2 }] },
      { name: "prefix", points: [{ x: 8, value: 3 }, { x: 16, value: 4 }] },
    ]);
    expect(toRows(s)).toEqual([{ x: 8, random: 1, prefix: 3 }, { x: 16, random: 2, prefix: 4 }]);
  });
});

describe("formatting", () => {
  it("formats numbers, shares and ratio ranges", () => {
    expect(fmt(454.98, 0)).toBe("455");
    expect(fmt(301026)).toBe("301,026");
    expect(fmt(NaN)).toBe("–");
    expect(pct(0.9935, 1)).toBe("99.4%");
    expect(pct(0.99, 1)).toBe("99.0%");
    expect(ratioRange([1.84, 2.31, 2.0])).toBe("1.8–2.3×");
  });
});

describe("niceTicks", () => {
  it("covers the data and zero with round steps", () => {
    expect(niceTicks(-38.6, -14.2)).toEqual([-40, -30, -20, -10, 0]);
    expect(niceTicks(0, 0.92)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(niceTicks(0, 414_000)).toEqual([0, 200_000, 400_000, 600_000]);
    expect(niceTicks(-41, 84)).toEqual([-50, 0, 50, 100]);
  });
});
