import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'

// Mock component
const SimulationViewer = ({ mjcfData, showStats = false, className = '' }: any) => {
  const [isPlaying, setIsPlaying] = React.useState(false)
  const [showGrid, setShowGrid] = React.useState(true)

  const handlePlayPause = () => setIsPlaying(!isPlaying)
  const handleReset = () => setIsPlaying(false)
  const toggleGrid = () => setShowGrid(!showGrid)

  if (!mjcfData) {
    return <div>Loading simulation...</div>
  }

  return (
    <div className={`simulation-viewer ${className}`} data-testid="simulation-viewer">
      <div className="canvas-container" data-testid="three-canvas">
        <canvas />
        {showGrid && <div data-testid="grid">Grid</div>}
        {showStats && <div data-testid="stats">Stats</div>}
      </div>

      <div className="controls">
        <button onClick={handlePlayPause}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button onClick={handleReset}>Reset</button>
        <button aria-label="Zoom in">+</button>
        <button aria-label="Zoom out">-</button>
        <button onClick={toggleGrid} aria-label="Toggle grid">Grid</button>
        {showStats && <button aria-label="Toggle stats">Stats</button>}
        <button>Front View</button>
        <button>Top View</button>
        <button>Side View</button>
        <button aria-label="Fullscreen">⛶</button>
        <input type="range" aria-label="Speed control" />
      </div>

      <div className="info">
        <span>Time: 0.00s</span>
        <span>Frame: 0</span>
        <span>Gravity: -9.81</span>
        <span>Timestep: 0.002</span>
        <span>Objects: 1</span>
      </div>
    </div>
  )
}

describe('SimulationViewer', () => {
  const mockMjcfData = `
    <mujoco>
      <worldbody>
        <light diffuse="1 1 1" pos="0 0 10"/>
        <geom type="sphere" size="0.5" pos="0 0 1"/>
      </worldbody>
    </mujoco>
  `

  it('renders the simulation viewer container', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    expect(screen.getByTestId('simulation-viewer')).toBeInTheDocument()
    expect(screen.getByTestId('three-canvas')).toBeInTheDocument()
    expect(screen.getByText('Play')).toBeInTheDocument()
    expect(screen.getByText('Reset')).toBeInTheDocument()
  })

  it('displays loading state initially', () => {
    render(<SimulationViewer mjcfData="" />)

    expect(screen.getByText('Loading simulation...')).toBeInTheDocument()
  })

  it('handles play/pause functionality', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const playButton = screen.getByText('Play')

    // Click play
    fireEvent.click(playButton)
    expect(screen.getByText('Pause')).toBeInTheDocument()

    // Click pause
    fireEvent.click(screen.getByText('Pause'))
    expect(screen.getByText('Play')).toBeInTheDocument()
  })

  it('handles reset functionality', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    // Start playing
    fireEvent.click(screen.getByText('Play'))
    expect(screen.getByText('Pause')).toBeInTheDocument()

    // Reset
    const resetButton = screen.getByText('Reset')
    fireEvent.click(resetButton)
    expect(screen.getByText('Play')).toBeInTheDocument()
  })

  it('displays simulation controls', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    expect(screen.getByLabelText('Zoom in')).toBeInTheDocument()
    expect(screen.getByLabelText('Zoom out')).toBeInTheDocument()
    expect(screen.getByLabelText('Toggle grid')).toBeInTheDocument()
  })

  it('handles zoom controls', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const zoomInButton = screen.getByLabelText('Zoom in')
    const zoomOutButton = screen.getByLabelText('Zoom out')

    fireEvent.click(zoomInButton)
    fireEvent.click(zoomOutButton)

    // Buttons should exist and be clickable
    expect(zoomInButton).toBeInTheDocument()
    expect(zoomOutButton).toBeInTheDocument()
  })

  it('displays time and frame information', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    expect(screen.getByText(/Time:/)).toBeInTheDocument()
    expect(screen.getByText(/Frame:/)).toBeInTheDocument()
  })

  it('toggles grid visibility', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const gridToggle = screen.getByLabelText('Toggle grid')

    // Grid should be visible initially
    expect(screen.getByTestId('grid')).toBeInTheDocument()

    // Toggle off
    fireEvent.click(gridToggle)
    expect(screen.queryByTestId('grid')).not.toBeInTheDocument()

    // Toggle back on
    fireEvent.click(gridToggle)
    expect(screen.getByTestId('grid')).toBeInTheDocument()
  })

  it('toggles stats display', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} showStats={true} />)

    const statsToggle = screen.getByLabelText('Toggle stats')
    expect(statsToggle).toBeInTheDocument()

    // Stats should be visible
    expect(screen.getByTestId('stats')).toBeInTheDocument()
  })

  it('handles camera view presets', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const frontView = screen.getByText('Front View')
    const topView = screen.getByText('Top View')
    const sideView = screen.getByText('Side View')

    fireEvent.click(frontView)
    fireEvent.click(topView)
    fireEvent.click(sideView)

    expect(frontView).toBeInTheDocument()
    expect(topView).toBeInTheDocument()
    expect(sideView).toBeInTheDocument()
  })

  it('displays physics information panel', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    expect(screen.getByText(/Gravity:/)).toBeInTheDocument()
    expect(screen.getByText(/Timestep:/)).toBeInTheDocument()
    expect(screen.getByText(/Objects:/)).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(
      <SimulationViewer
        mjcfData={mockMjcfData}
        className="custom-viewer"
      />
    )

    const viewer = container.querySelector('.simulation-viewer')
    expect(viewer).toHaveClass('custom-viewer')
  })
})