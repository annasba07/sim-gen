import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SketchCanvas } from '@/components/sketch-canvas'

// Mock canvas context
const mockGetContext = jest.fn()
const mockToDataURL = jest.fn(() => 'data:image/png;base64,mockData')

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = mockGetContext
  HTMLCanvasElement.prototype.toDataURL = mockToDataURL

  mockGetContext.mockReturnValue({
    clearRect: jest.fn(),
    strokeRect: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
    fillRect: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    drawImage: jest.fn(),
    getImageData: jest.fn(() => ({ data: [] })),
    putImageData: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    scale: jest.fn(),
    lineCap: 'round',
    lineJoin: 'round',
    lineWidth: 2,
    strokeStyle: '#000000',
    fillStyle: '#ffffff',
  })
})

describe('SketchCanvas', () => {
  const mockOnChange = jest.fn()
  const mockOnClear = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders the canvas and controls', () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    expect(screen.getByRole('img', { hidden: true })).toBeInTheDocument() // Canvas is treated as img role
    expect(screen.getByText(/clear/i)).toBeInTheDocument()
    expect(screen.getByText(/pen/i)).toBeInTheDocument()
    expect(screen.getByText(/eraser/i)).toBeInTheDocument()
  })

  it('initializes canvas context on mount', () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    expect(mockGetContext).toHaveBeenCalledWith('2d')
  })

  it('handles drawing on canvas', async () => {
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')
    expect(canvas).toBeInTheDocument()

    // Simulate mouse events for drawing
    if (canvas) {
      fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 })
      fireEvent.mouseMove(canvas, { clientX: 150, clientY: 150 })
      fireEvent.mouseUp(canvas)
    }

    // Wait for debounced onChange
    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled()
    }, { timeout: 1000 })
  })

  it('handles touch events for mobile drawing', async () => {
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')

    if (canvas) {
      fireEvent.touchStart(canvas, {
        touches: [{ clientX: 100, clientY: 100 }]
      })
      fireEvent.touchMove(canvas, {
        touches: [{ clientX: 150, clientY: 150 }]
      })
      fireEvent.touchEnd(canvas)
    }

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled()
    }, { timeout: 1000 })
  })

  it('switches between pen and eraser tools', async () => {
    const user = userEvent.setup()
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const eraserButton = screen.getByText(/eraser/i)
    const penButton = screen.getByText(/pen/i)

    // Switch to eraser
    await user.click(eraserButton)
    expect(eraserButton.parentElement).toHaveClass('bg-primary')

    // Switch back to pen
    await user.click(penButton)
    expect(penButton.parentElement).toHaveClass('bg-primary')
  })

  it('changes pen size', async () => {
    const user = userEvent.setup()
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const sizeSlider = container.querySelector('input[type="range"]')
    expect(sizeSlider).toBeInTheDocument()

    if (sizeSlider) {
      fireEvent.change(sizeSlider, { target: { value: '10' } })
      expect(sizeSlider).toHaveValue('10')
    }
  })

  it('changes pen color', async () => {
    const user = userEvent.setup()
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const colorPicker = container.querySelector('input[type="color"]')
    expect(colorPicker).toBeInTheDocument()

    if (colorPicker) {
      fireEvent.change(colorPicker, { target: { value: '#ff0000' } })
      expect(colorPicker).toHaveValue('#ff0000')
    }
  })

  it('clears the canvas', async () => {
    const user = userEvent.setup()
    const mockClearRect = jest.fn()
    mockGetContext.mockReturnValue({
      ...mockGetContext(),
      clearRect: mockClearRect,
    })

    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const clearButton = screen.getByText(/clear/i)
    await user.click(clearButton)

    expect(mockClearRect).toHaveBeenCalled()
    expect(mockOnClear).toHaveBeenCalled()
    expect(mockOnChange).toHaveBeenCalledWith('')
  })

  it('undoes the last drawing action', async () => {
    const user = userEvent.setup()
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const undoButton = screen.getByLabelText(/undo/i)
    expect(undoButton).toBeInTheDocument()

    // Initially disabled when no history
    expect(undoButton).toBeDisabled()

    // Simulate drawing to create history
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')
    if (canvas) {
      fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 })
      fireEvent.mouseUp(canvas)
    }

    // Now undo should be enabled
    await waitFor(() => {
      expect(undoButton).not.toBeDisabled()
    })

    await user.click(undoButton)
    expect(mockOnChange).toHaveBeenCalled()
  })

  it('redoes a previously undone action', async () => {
    const user = userEvent.setup()
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const redoButton = screen.getByLabelText(/redo/i)
    expect(redoButton).toBeInTheDocument()

    // Initially disabled
    expect(redoButton).toBeDisabled()
  })

  it('exports canvas as image data', async () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    // Draw something
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')
    if (canvas) {
      fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 })
      fireEvent.mouseMove(canvas, { clientX: 150, clientY: 150 })
      fireEvent.mouseUp(canvas)
    }

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith('data:image/png;base64,mockData')
    })
  })

  it('handles canvas resize on window resize', () => {
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')
    const initialWidth = canvas?.width

    // Trigger window resize
    global.innerWidth = 1024
    global.innerHeight = 768
    fireEvent(window, new Event('resize'))

    // Canvas should adjust to new dimensions
    expect(canvas?.width).toBeDefined()
  })

  it('prevents default touch behavior to avoid scrolling while drawing', () => {
    const { container } = render(
      <SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />
    )

    const canvas = container.querySelector('canvas')

    if (canvas) {
      const touchEvent = new TouchEvent('touchstart', {
        touches: [{ clientX: 100, clientY: 100 } as Touch]
      })

      const preventDefault = jest.spyOn(touchEvent, 'preventDefault')
      fireEvent(canvas, touchEvent)

      expect(preventDefault).toHaveBeenCalled()
    }
  })

  it('displays loading state when specified', () => {
    render(
      <SketchCanvas
        onChange={mockOnChange}
        onClear={mockOnClear}
        isLoading={true}
      />
    )

    expect(screen.getByText(/processing/i)).toBeInTheDocument()
  })

  it('disables controls when disabled prop is true', () => {
    render(
      <SketchCanvas
        onChange={mockOnChange}
        onClear={mockOnClear}
        disabled={true}
      />
    )

    const clearButton = screen.getByText(/clear/i)
    expect(clearButton).toBeDisabled()

    const penButton = screen.getByText(/pen/i)
    expect(penButton).toBeDisabled()
  })

  it('applies custom className when provided', () => {
    const { container } = render(
      <SketchCanvas
        onChange={mockOnChange}
        onClear={mockOnClear}
        className="custom-class"
      />
    )

    const wrapper = container.firstChild
    expect(wrapper).toHaveClass('custom-class')
  })

  it('handles save functionality', async () => {
    const user = userEvent.setup()
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const saveButton = screen.getByLabelText(/save/i)

    if (saveButton) {
      await user.click(saveButton)

      // Should trigger download
      expect(mockToDataURL).toHaveBeenCalled()
    }
  })

  it('supports keyboard shortcuts', async () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    // Test Ctrl+Z for undo
    fireEvent.keyDown(document, { key: 'z', ctrlKey: true })

    // Test Ctrl+Y for redo
    fireEvent.keyDown(document, { key: 'y', ctrlKey: true })

    // Test Delete for clear
    fireEvent.keyDown(document, { key: 'Delete' })
    expect(mockOnClear).toHaveBeenCalled()
  })
})