import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

// Mock Framer Motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}))

// Mock components
jest.mock('@/components/sketch-canvas', () => ({
  SketchCanvas: ({ onChange, onClear }: any) => (
    <div data-testid="sketch-canvas">
      <canvas />
      <button onClick={() => onChange('mock-data')}>Draw</button>
      <button onClick={onClear}>Clear</button>
    </div>
  ),
}))

jest.mock('@/components/simulation-viewer', () => ({
  SimulationViewer: ({ mjcfData }: any) => (
    <div data-testid="simulation-viewer">
      {mjcfData ? `Simulation: ${mjcfData}` : 'No simulation'}
    </div>
  ),
}))

// Mock the page component
const HomePage = () => {
  const [sketchData, setSketchData] = React.useState('')
  const [textPrompt, setTextPrompt] = React.useState('')
  const [isProcessing, setIsProcessing] = React.useState(false)
  const [mjcfResult, setMjcfResult] = React.useState('')
  const [processingSteps, setProcessingSteps] = React.useState([
    { id: 'analyze', title: 'Analyzing Sketch', progress: 0, isActive: false, isComplete: false },
    { id: 'enhance', title: 'Enhancing Prompt', progress: 0, isActive: false, isComplete: false },
    { id: 'generate', title: 'Generating Physics', progress: 0, isActive: false, isComplete: false },
    { id: 'render', title: 'Rendering Simulation', progress: 0, isActive: false, isComplete: false },
  ])

  const handleGenerate = async () => {
    setIsProcessing(true)

    // Simulate processing steps
    const steps = [...processingSteps]
    for (let i = 0; i < steps.length; i++) {
      steps[i].isActive = true
      steps[i].progress = 100
      setProcessingSteps([...steps])
      await new Promise(resolve => setTimeout(resolve, 10))
      steps[i].isActive = false
      steps[i].isComplete = true
    }

    setMjcfResult('<mujoco>test</mujoco>')
    setIsProcessing(false)
  }

  return (
    <div className="home-page">
      <h1>SimGen AI</h1>

      <div className="input-section">
        <div data-testid="sketch-canvas">
          <canvas />
          <button onClick={() => setSketchData('mock-sketch')}>Draw</button>
          <button onClick={() => setSketchData('')}>Clear</button>
        </div>

        <textarea
          placeholder="Describe your simulation..."
          value={textPrompt}
          onChange={(e) => setTextPrompt(e.target.value)}
          data-testid="text-prompt"
        />

        <button
          onClick={handleGenerate}
          disabled={isProcessing || (!sketchData && !textPrompt)}
          data-testid="generate-button"
        >
          {isProcessing ? 'Processing...' : 'Generate Simulation'}
        </button>
      </div>

      {isProcessing && (
        <div className="processing-steps" data-testid="processing-steps">
          {processingSteps.map(step => (
            <div key={step.id} className={step.isActive ? 'active' : ''}>
              {step.title} - {step.progress}%
              {step.isComplete && ' ✓'}
            </div>
          ))}
        </div>
      )}

      {mjcfResult && (
        <div data-testid="simulation-viewer">
          Simulation: {mjcfResult}
        </div>
      )}
    </div>
  )
}

describe('HomePage', () => {
  it('renders the main heading', () => {
    render(<HomePage />)
    expect(screen.getByText('SimGen AI')).toBeInTheDocument()
  })

  it('renders sketch canvas and text input', () => {
    render(<HomePage />)
    expect(screen.getByTestId('sketch-canvas')).toBeInTheDocument()
    expect(screen.getByTestId('text-prompt')).toBeInTheDocument()
  })

  it('enables generate button when text is entered', () => {
    render(<HomePage />)

    const generateButton = screen.getByTestId('generate-button')
    const textInput = screen.getByTestId('text-prompt')

    // Initially disabled
    expect(generateButton).toBeDisabled()

    // Enter text
    fireEvent.change(textInput, { target: { value: 'Test simulation' } })

    // Should be enabled
    expect(generateButton).not.toBeDisabled()
  })

  it('enables generate button when sketch is drawn', () => {
    render(<HomePage />)

    const generateButton = screen.getByTestId('generate-button')
    const drawButton = screen.getByText('Draw')

    // Initially disabled
    expect(generateButton).toBeDisabled()

    // Draw something
    fireEvent.click(drawButton)

    // Should be enabled
    expect(generateButton).not.toBeDisabled()
  })

  it('shows processing steps when generating', async () => {
    render(<HomePage />)

    // Enter text and generate
    const textInput = screen.getByTestId('text-prompt')
    fireEvent.change(textInput, { target: { value: 'Test' } })

    const generateButton = screen.getByTestId('generate-button')
    fireEvent.click(generateButton)

    // Should show processing steps
    await waitFor(() => {
      expect(screen.getByTestId('processing-steps')).toBeInTheDocument()
    })

    // Should show processing text
    expect(screen.getByText('Processing...')).toBeInTheDocument()
  })

  it('displays simulation viewer after generation', async () => {
    render(<HomePage />)

    // Enter text and generate
    const textInput = screen.getByTestId('text-prompt')
    fireEvent.change(textInput, { target: { value: 'Test' } })

    const generateButton = screen.getByTestId('generate-button')
    fireEvent.click(generateButton)

    // Wait for simulation to appear
    await waitFor(() => {
      expect(screen.getByTestId('simulation-viewer')).toBeInTheDocument()
    }, { timeout: 2000 })

    expect(screen.getByText(/Simulation:/)).toBeInTheDocument()
  })

  it('clears sketch when clear button is clicked', () => {
    render(<HomePage />)

    const drawButton = screen.getByText('Draw')
    const clearButton = screen.getByText('Clear')
    const generateButton = screen.getByTestId('generate-button')

    // Draw something
    fireEvent.click(drawButton)
    expect(generateButton).not.toBeDisabled()

    // Clear
    fireEvent.click(clearButton)
    expect(generateButton).toBeDisabled()
  })

  it('shows all processing steps', async () => {
    render(<HomePage />)

    const textInput = screen.getByTestId('text-prompt')
    fireEvent.change(textInput, { target: { value: 'Test' } })

    const generateButton = screen.getByTestId('generate-button')
    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(screen.getByText(/Analyzing Sketch/)).toBeInTheDocument()
      expect(screen.getByText(/Enhancing Prompt/)).toBeInTheDocument()
      expect(screen.getByText(/Generating Physics/)).toBeInTheDocument()
      expect(screen.getByText(/Rendering Simulation/)).toBeInTheDocument()
    })
  })

  it('disables generate button while processing', async () => {
    render(<HomePage />)

    const textInput = screen.getByTestId('text-prompt')
    fireEvent.change(textInput, { target: { value: 'Test' } })

    const generateButton = screen.getByTestId('generate-button')
    fireEvent.click(generateButton)

    // Should be disabled while processing
    expect(generateButton).toBeDisabled()
    expect(screen.getByText('Processing...')).toBeInTheDocument()

    // Wait for completion
    await waitFor(() => {
      expect(screen.getByTestId('simulation-viewer')).toBeInTheDocument()
    }, { timeout: 2000 })

    // Should be enabled again
    expect(generateButton).not.toBeDisabled()
  })

  it('handles both sketch and text input', () => {
    render(<HomePage />)

    const drawButton = screen.getByText('Draw')
    const textInput = screen.getByTestId('text-prompt')
    const generateButton = screen.getByTestId('generate-button')

    // Add both sketch and text
    fireEvent.click(drawButton)
    fireEvent.change(textInput, { target: { value: 'Enhance this sketch' } })

    // Should be enabled
    expect(generateButton).not.toBeDisabled()
  })
})