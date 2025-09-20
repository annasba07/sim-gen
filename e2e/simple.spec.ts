import { test, expect } from '@playwright/test'

test.describe('Simple E2E Test', () => {
  test('should navigate to the homepage', async ({ page }) => {
    // Navigate to the application
    await page.goto('/')

    // Check that the page loads
    await expect(page).toHaveTitle(/SimGen/)

    // Check for main heading
    const heading = page.locator('h1')
    await expect(heading).toBeVisible()
  })

  test('should have a canvas element', async ({ page }) => {
    await page.goto('/')

    // Wait for canvas to be visible
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()
  })
})