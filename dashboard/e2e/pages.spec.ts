import { expect, test } from "@playwright/test";

const PAGES = [
  ["overview", "Serving LLM agents on one Mac"],
  ["engines", "Engines"],
  ["routing", "Routing"],
  ["provider", "Provider cache"],
  ["lifetimes", "Cache lifetime"],
  ["reuse", "What buys the most reuse"],
  ["simulator", "Simulator vs a real engine"],
  ["provenance", "Provenance and scope"],
] as const;

for (const scheme of ["light", "dark"] as const) {
  test.describe(`${scheme} mode`, () => {
    test.use({ colorScheme: scheme });
    for (const [id, heading] of PAGES) {
      test(`${id} renders from the snapshot`, async ({ page }) => {
        const errors: string[] = [];
        page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
        page.on("pageerror", (e) => errors.push(e.message));
        await page.goto(`/#/${id}`);
        await expect(page.getByRole("heading", { level: 2, name: heading, exact: true })).toBeVisible();
        await expect(page.getByRole("link", { name: /Overview/ })).toBeVisible();
        if (id !== "provenance") {
          await expect(page.locator("svg.recharts-surface").first()).toBeVisible();
        }
        expect(errors).toEqual([]);
      });
    }
  });
}

test("table view toggles", async ({ page }) => {
  await page.goto("/#/routing");
  await page.getByRole("button", { name: "Table" }).first().click();
  await expect(page.getByRole("columnheader", { name: "Value" }).first()).toBeVisible();
});

test("theme switch overrides the system setting", async ({ page }) => {
  await page.goto("/#/overview");
  await page.getByRole("radio", { name: "dark" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("radio", { name: "system" }).click();
  await expect(page.locator("html")).not.toHaveAttribute("data-theme", /.+/);
});
