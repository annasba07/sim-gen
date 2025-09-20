import { test, expect, Page } from '@playwright/test'

test.describe('Sketch to Simulation Flow', () => {
  let page: Page

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage
    await page.goto('/')
    // Wait for app to load
    await page.waitForLoadState('networkidle')
  })

  test('should load the application', async () => {
    // Check main elements are present
    await expect(page.locator('h1:has-text("SimGen AI")')).toBeVisible()
    await expect(page.locator('canvas')).toBeVisible()
    await expect(page.locator('button:has-text("Generate Simulation")')).toBeVisible()
  })

  test('should draw on canvas and generate simulation', async () => {
    // Get canvas element
    const canvas = page.locator('canvas').first()
    const box = await canvas.boundingBox()

    if (!box) {
      throw new Error('Canvas not found')
    }

    // Draw a simple pendulum shape
    await page.mouse.move(box.x + box.width / 2, box.y + 50)
    await page.mouse.down()
    await page.mouse.move(box.x + box.width / 2, box.y + 200, { steps: 10 })
    await page.mouse.up()

    // Draw a circle for the pendulum bob
    await page.mouse.move(box.x + box.width / 2 - 20, box.y + 200)
    await page.mouse.down()

    // Draw circle
    for (let angle = 0; angle < Math.PI * 2; angle += 0.1) {
      const x = box.x + box.width / 2 + Math.cos(angle) * 20
      const y = box.y + 200 + Math.sin(angle) * 20
      await page.mouse.move(x, y)
    }
    await page.mouse.up()

    // Add text description
    await page.fill('textarea[placeholder*="Describe"]', 'A simple pendulum with gravity')

    // Generate simulation
    await page.click('button:has-text("Generate Simulation")')

    // Wait for processing to start
    await expect(page.locator('text=Analyzing Sketch')).toBeVisible({ timeout: 10000 })

    // Wait for simulation to be generated
    await expect(page.locator('[data-testid="simulation-viewer"]')).toBeVisible({ timeout: 30000 })

    // Verify 3D viewer is loaded
    await expect(page.locator('canvas').nth(1)).toBeVisible() // Second canvas is Three.js
  })

  test('should clear canvas', async () => {
    const canvas = page.locator('canvas').first()
    const box = await canvas.boundingBox()

    if (!box) {
      throw new Error('Canvas not found')
    }

    // Draw something
    await page.mouse.move(box.x + 100, box.y + 100)
    await page.mouse.down()
    await page.mouse.move(box.x + 200, box.y + 200)
    await page.mouse.up()

    // Clear canvas
    await page.click('button:has-text("Clear")')

    // Canvas should be cleared (check via screenshot or canvas data)
    const canvasData = await page.evaluate(() => {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement
      const ctx = canvas?.getContext('2d')
      const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height)
      // Check if canvas is mostly empty (white/transparent)
      return imageData?.data.filter(pixel => pixel !== 0 && pixel !== 255).length
    })

    expect(canvasData).toBeLessThan(100) // Should be mostly empty
  })

  test('should switch between pen and eraser', async () => {
    // Select eraser
    await page.click('button:has-text("Eraser")')
    await expect(page.locator('button:has-text("Eraser")')).toHaveClass(/bg-primary/)

    // Switch back to pen
    await page.click('button:has-text("Pen")')
    await expect(page.locator('button:has-text("Pen")')).toHaveClass(/bg-primary/)
  })

  test('should change pen color and size', async () => {
    // Change color
    const colorPicker = page.locator('input[type="color"]')
    await colorPicker.fill('#ff0000')

    // Change size
    const sizeSlider = page.locator('input[type="range"]')
    await sizeSlider.fill('10')

    // Draw with new settings
    const canvas = page.locator('canvas').first()
    const box = await canvas.boundingBox()

    if (box) {
      await page.mouse.move(box.x + 100, box.y + 100)
      await page.mouse.down()
      await page.mouse.move(box.x + 200, box.y + 100)
      await page.mouse.up()
    }
  })

  test('should handle text-only prompt', async () => {
    // Don't draw anything, just enter text
    await page.fill('textarea[placeholder*="Describe"]', 'A robotic arm with 6 degrees of freedom picking up colored balls')

    // Generate simulation
    await page.click('button:has-text("Generate Simulation")')

    // Should still process and generate
    await expect(page.locator('text=Enhancing Prompt')).toBeVisible({ timeout: 10000 })
    await expect(page.locator('[data-testid="simulation-viewer"]')).toBeVisible({ timeout: 30000 })
  })

  test('should display processing steps', async () => {
    // Add description
    await page.fill('textarea[placeholder*="Describe"]', 'Bouncing ball simulation')

    // Generate
    await page.click('button:has-text("Generate Simulation")')

    // Check processing steps appear in order
    const steps = [
      'Analyzing Sketch',
      'Enhancing Prompt',
      'Generating Physics',
      'Rendering Simulation'
    ]

    for (const step of steps) {
      await expect(page.locator(`text=${step}`)).toBeVisible({ timeout: 15000 })
    }
  })

  test('should control simulation playback', async () => {
    // Generate a simple simulation first
    await page.fill('textarea[placeholder*="Describe"]', 'Simple pendulum')
    await page.click('button:has-text("Generate Simulation")')

    // Wait for simulation
    await page.waitForSelector('[data-testid="simulation-viewer"]', { timeout: 30000 })

    // Play simulation
    await page.click('button:has-text("Play")')
    await expect(page.locator('button:has-text("Pause")')).toBeVisible()

    // Pause simulation
    await page.click('button:has-text("Pause")')
    await expect(page.locator('button:has-text("Play")')).toBeVisible()

    // Reset simulation
    await page.click('button:has-text("Reset")')
  })

  test('should handle zoom controls', async () => {
    // Generate simulation
    await page.fill('textarea[placeholder*="Describe"]', 'Simple cube')
    await page.click('button:has-text("Generate Simulation")')
    await page.waitForSelector('[data-testid="simulation-viewer"]', { timeout: 30000 })

    // Zoom in
    await page.click('button[aria-label="Zoom in"]')

    // Zoom out
    await page.click('button[aria-label="Zoom out"]')

    // Reset view
    await page.click('button:has-text("Reset View")')
  })

  test('should toggle grid and stats', async () => {
    // Generate simulation
    await page.fill('textarea[placeholder*="Describe"]', 'Simple sphere')
    await page.click('button:has-text("Generate Simulation")')
    await page.waitForSelector('[data-testid="simulation-viewer"]', { timeout: 30000 })

    // Toggle grid
    await page.click('button[aria-label="Toggle grid"]')

    // Toggle stats
    if (await page.locator('button[aria-label="Toggle stats"]').isVisible()) {
      await page.click('button[aria-label="Toggle stats"]')
    }
  })

  test('should handle keyboard shortcuts', async () => {
    await page.goto('/')

    // Test keyboard shortcuts
    await page.keyboard.press('Control+z') // Undo
    await page.keyboard.press('Control+y') // Redo
    await page.keyboard.press('Delete') // Clear canvas

    // In simulation view
    await page.fill('textarea[placeholder*="Describe"]', 'Test object')
    await page.click('button:has-text("Generate Simulation")')
    await page.waitForSelector('[data-testid="simulation-viewer"]', { timeout: 30000 })

    await page.keyboard.press('Space') // Play/Pause
    await page.keyboard.press('r') // Reset
    await page.keyboard.press('g') // Toggle grid
  })

  test('should handle API errors gracefully', async () => {
    // Mock API error
    await page.route('**/api/v1/simulation/generate', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal server error' })
      })
    })

    await page.fill('textarea[placeholder*="Describe"]', 'Test simulation')
    await page.click('button:has-text("Generate Simulation")')

    // Should show error message
    await expect(page.locator('text=Error generating simulation')).toBeVisible({ timeout: 10000 })
  })

  test('should work on mobile viewport', async ({ browser }) => {
    // Create mobile context
    const context = await browser.newContext({
      viewport: { width: 375, height: 667 },
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
    })

    const mobilePage = await context.newPage()
    await mobilePage.goto('/')

    // Check mobile layout
    await expect(mobilePage.locator('canvas')).toBeVisible()
    await expect(mobilePage.locator('button:has-text("Generate")')).toBeVisible()

    // Touch drawing
    const canvas = mobilePage.locator('canvas').first()
    const box = await canvas.boundingBox()

    if (box) {
      await mobilePage.touchscreen.tap(box.x + 100, box.y + 100)
      await mobilePage.waitForTimeout(100)
      await mobilePage.touchscreen.tap(box.x + 200, box.y + 200)
    }

    await context.close()
  })

  test('should save and load sketches', async () => {
    // Draw something
    const canvas = page.locator('canvas').first()
    const box = await canvas.boundingBox()

    if (box) {
      await page.mouse.move(box.x + 100, box.y + 100)
      await page.mouse.down()
      await page.mouse.move(box.x + 200, box.y + 200)
      await page.mouse.up()
    }

    // Save sketch
    if (await page.locator('button[aria-label="Save sketch"]').isVisible()) {
      const downloadPromise = page.waitForEvent('download')
      await page.click('button[aria-label="Save sketch"]')
      const download = await downloadPromise
      expect(download.suggestedFilename()).toContain('sketch')
    }
  })

  test('should handle concurrent requests', async () => {
    // Open multiple tabs
    const context = page.context()
    const page2 = await context.newPage()
    await page2.goto('/')

    // Generate simulations in both tabs
    await page.fill('textarea[placeholder*="Describe"]', 'Simulation 1')
    await page2.fill('textarea[placeholder*="Describe"]', 'Simulation 2')

    await Promise.all([
      page.click('button:has-text("Generate Simulation")'),
      page2.click('button:has-text("Generate Simulation")')
    ])

    // Both should complete
    await expect(page.locator('[data-testid="simulation-viewer"]')).toBeVisible({ timeout: 30000 })
    await expect(page2.locator('[data-testid="simulation-viewer"]')).toBeVisible({ timeout: 30000 })

    await page2.close()
  })

  test('should export simulation', async () => {
    // Generate simulation
    await page.fill('textarea[placeholder*="Describe"]', 'Export test')
    await page.click('button:has-text("Generate Simulation")')
    await page.waitForSelector('[data-testid="simulation-viewer"]', { timeout: 30000 })

    // Export
    if (await page.locator('button[aria-label="Export"]').isVisible()) {
      const downloadPromise = page.waitForEvent('download')
      await page.click('button[aria-label="Export"]')
      const download = await downloadPromise
      expect(download.suggestedFilename()).toMatch(/\.(xml|mjcf)$/)
    }
  })
})