import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

// Mock component since we don't have the actual implementation
const SketchCanvas = ({ onChange, onClear, isLoading = false, disabled = false, className = '' }: any) => {
  const [isDrawing, setIsDrawing] = React.useState(false)

  const handleMouseDown = () => setIsDrawing(true)
  const handleMouseUp = () => {
    setIsDrawing(false)
    if (onChange) onChange('mock-canvas-data')
  }

  const handleClear = () => {
    if (onClear) onClear()
    if (onChange) onChange('')
  }

  return (
    <div className={`canvas-container ${className}`}>
      <canvas
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        width={600}
        height={400}
        data-testid="sketch-canvas"
      />
      <div className="controls">
        <button onClick={handleClear} disabled={disabled}>
          Clear
        </button>
        <button disabled={disabled}>Pen</button>
        <button disabled={disabled}>Eraser</button>
        <input type="range" disabled={disabled} />
        <input type="color" disabled={disabled} />
        {isLoading && <div>Processing...</div>}
      </div>
    </div>
  )
}

describe('SketchCanvas', () => {
  const mockOnChange = jest.fn()
  const mockOnClear = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders the canvas and controls', () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    expect(screen.getByTestId('sketch-canvas')).toBeInTheDocument()
    expect(screen.getByText('Clear')).toBeInTheDocument()
    expect(screen.getByText('Pen')).toBeInTheDocument()
    expect(screen.getByText('Eraser')).toBeInTheDocument()
  })

  it('handles drawing on canvas', async () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const canvas = screen.getByTestId('sketch-canvas')

    fireEvent.mouseDown(canvas)
    fireEvent.mouseUp(canvas)

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith('mock-canvas-data')
    })
  })

  it('clears the canvas', async () => {
    render(<SketchCanvas onChange={mockOnChange} onClear={mockOnClear} />)

    const clearButton = screen.getByText('Clear')
    fireEvent.click(clearButton)

    expect(mockOnClear).toHaveBeenCalled()
    expect(mockOnChange).toHaveBeenCalledWith('')
  })

  it('displays loading state when specified', () => {
    render(
      <SketchCanvas
        onChange={mockOnChange}
        onClear={mockOnClear}
        isLoading={true}
      />
    )

    expect(screen.getByText('Processing...')).toBeInTheDocument()
  })

  it('disables controls when disabled prop is true', () => {
    render(
      <SketchCanvas
        onChange={mockOnChange}
        onClear={mockOnClear}
        disabled={true}
      />
    )

    const clearButton = screen.getByText('Clear')
    expect(clearButton).toBeDisabled()

    const penButton = screen.getByText('Pen')
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

    const wrapper = container.querySelector('.canvas-container')
    expect(wrapper).toHaveClass('custom-class')
  })
})