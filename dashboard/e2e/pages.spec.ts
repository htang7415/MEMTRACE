import { expect, test } from "@playwright/test";

const PAGES = [
  ["overview", "Serving LLM agents efficiently"],
  ["system", "How MEMTRACE is built"],
  ["engines", "Which engine and cache settings serve the GPU tier?"],
  ["routing", "How should agent traffic be routed, and when should it overflow?"],
  ["replay", "Does a routing policy win on real agent traffic?"],
  ["context", "What should an agent send each step, and can the gateway manage it?"],
  ["lifetime", "How long should agent KV be kept?"],
  ["placement", "What buys the most KV reuse at fleet scale?"],
  ["paused", "What does an SSD KV tier save agent sessions that pause?"],
  ["methods", "How the numbers are produced"],
] as const;
const NO_CHARTS = new Set(["overview", "system", "methods"]);

for (const scheme of ["light", "dark"] as const) {
  test.describe(`${scheme} mode`, () => {
    test.use({ colorScheme: scheme });
    for (const [id, heading] of PAGES) {
      test(`${id} renders from saved results`, async ({ page }) => {
        const errors: string[] = [];
        page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto(`/#/${id}`);
        await expect(page.getByRole("heading", { level: 2, name: heading })).toBeVisible();
        await expect(page.getByRole("link", { name: /Overview/ })).toBeVisible();
        if (!NO_CHARTS.has(id)) {
          await expect(page.locator("svg.recharts-surface").first()).toBeVisible();
        }
        expect(errors).toEqual([]);
      });
    }
  });
}

test("table view toggles and shows CIs", async ({ page }) => {
  await page.goto("/#/routing");
  await page.getByRole("button", { name: "Table" }).first().click();
  await expect(page.getByRole("columnheader", { name: "95% CI" }).first()).toBeVisible();
});

test("provenance lists every exported run", async ({ page }) => {
  const index = await (await page.request.get("/data/index.json")).json();
  await page.goto("/#/methods");
  for (const e of index.experiments) {
    await expect(page.getByRole("cell", { name: e.name, exact: true })).toBeVisible();
  }
});

test("theme switch overrides the system setting", async ({ page }) => {
  await page.goto("/#/overview");
  await page.getByRole("radio", { name: "dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("radio", { name: "system" }).click();
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
});

test("links to merged pages still land on their content", async ({ page }) => {
  for (const [old, heading] of [["provenance", "How the numbers are produced"], ["simulator", "What buys the most KV reuse at fleet scale?"],
    ["resume", "What does an SSD KV tier save agent sessions that pause?"]] as const) {
    await page.goto(`/#/${old}`);
    await expect(page.getByRole("heading", { level: 2, name: heading })).toBeVisible();
  }
});

test("overview shows the workflow, the system and the takeaways", async ({ page }) => {
  await page.goto("/#/overview");
  for (const name of ["How a result is made", "The system", "Key takeaways", "What did not work", "Technology"]) {
    await expect(page.getByRole("heading", { level: 3, name })).toBeVisible();
  }
  await expect(page.getByRole("figure", { name: "System architecture" })).toBeVisible();
});
