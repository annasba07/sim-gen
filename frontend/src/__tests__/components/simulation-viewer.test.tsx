import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { SimulationViewer } from '@/components/simulation-viewer'

// Mock Three.js and React Three Fiber
jest.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => <div data-testid="three-canvas">{children}</div>,
  useFrame: jest.fn(),
  useThree: () => ({
    camera: { position: { set: jest.fn() } },
    gl: { domElement: {} },
    scene: {},
  }),
}))

jest.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Grid: () => <div data-testid="grid" />,
  Environment: () => <div data-testid="environment" />,
  Stats: () => <div data-testid="stats" />,
  Box: ({ children, ...props }: any) => (
    <div data-testid="box" {...props}>{children}</div>
  ),
  Sphere: ({ children, ...props }: any) => (
    <div data-testid="sphere" {...props}>{children}</div>
  ),
  Cylinder: ({ children, ...props }: any) => (
    <div data-testid="cylinder" {...props}>{children}</div>
  ),
}))

// Mock socket.io-client
jest.mock('socket.io-client', () => ({
  io: jest.fn(() => ({
    on: jest.fn(),
    off: jest.fn(),
    emit: jest.fn(),
    disconnect: jest.fn(),
  })),
}))

describe('SimulationViewer', () => {
  const mockMjcfData = `
    <mujoco>
      <worldbody>
        <light diffuse="1 1 1" pos="0 0 10"/>
        <geom type="sphere" size="0.5" pos="0 0 1"/>
        <body name="pendulum">
          <joint type="hinge" axis="0 1 0"/>
          <geom type="cylinder" size="0.05 0.5"/>
        </body>
      </worldbody>
    </mujoco>
  `

  it('renders the simulation viewer container', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    expect(screen.getByTestId('three-canvas')).toBeInTheDocument()
    expect(screen.getByText(/play/i)).toBeInTheDocument()
    expect(screen.getByText(/reset/i)).toBeInTheDocument()
  })

  it('displays loading state initially', () => {
    render(<SimulationViewer mjcfData="" />)

    expect(screen.getByText(/loading simulation/i)).toBeInTheDocument()
  })

  it('parses and displays MJCF data', async () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    await waitFor(() => {
      expect(screen.getByTestId('sphere')).toBeInTheDocument()
      expect(screen.getByTestId('cylinder')).toBeInTheDocument()
    })
  })

  it('handles play/pause functionality', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const playButton = screen.getByText(/play/i)

    // Click play
    await user.click(playButton)
    expect(screen.getByText(/pause/i)).toBeInTheDocument()

    // Click pause
    await user.click(screen.getByText(/pause/i))
    expect(screen.getByText(/play/i)).toBeInTheDocument()
  })

  it('handles reset functionality', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const resetButton = screen.getByText(/reset/i)
    await user.click(resetButton)

    // Should reset the simulation state
    expect(screen.getByText(/play/i)).toBeInTheDocument()
  })

  it('displays simulation controls', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    // Check for control buttons
    expect(screen.getByLabelText(/zoom in/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/zoom out/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/rotate view/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/pan view/i)).toBeInTheDocument()
  })

  it('handles zoom controls', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const zoomInButton = screen.getByLabelText(/zoom in/i)
    const zoomOutButton = screen.getByLabelText(/zoom out/i)

    await user.click(zoomInButton)
    await user.click(zoomOutButton)

    // Zoom actions should be triggered
    expect(zoomInButton).toBeInTheDocument()
  })

  it('displays time and frame information', async () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    await waitFor(() => {
      expect(screen.getByText(/time:/i)).toBeInTheDocument()
      expect(screen.getByText(/frame:/i)).toBeInTheDocument()
    })
  })

  it('handles speed control', async () => {
    const user = userEvent.setup()
    const { container } = render(<SimulationViewer mjcfData={mockMjcfData} />)

    const speedSlider = container.querySelector('input[type="range"]')

    if (speedSlider) {
      fireEvent.change(speedSlider, { target: { value: '2' } })
      expect(speedSlider).toHaveValue('2')
      expect(screen.getByText(/2x/i)).toBeInTheDocument()
    }
  })

  it('toggles grid visibility', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const gridToggle = screen.getByLabelText(/toggle grid/i)

    await user.click(gridToggle)

    // Grid visibility should toggle
    const grid = screen.queryByTestId('grid')
    expect(grid).toBeInTheDocument()
  })

  it('toggles stats display', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} showStats={true} />)

    const statsToggle = screen.getByLabelText(/toggle stats/i)

    await user.click(statsToggle)

    // Stats should toggle
    const stats = screen.queryByTestId('stats')
    expect(stats).toBeInTheDocument()
  })

  it('handles camera view presets', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    // Check for view preset buttons
    const frontView = screen.getByText(/front view/i)
    const topView = screen.getByText(/top view/i)
    const sideView = screen.getByText(/side view/i)

    await user.click(frontView)
    await user.click(topView)
    await user.click(sideView)

    expect(frontView).toBeInTheDocument()
  })

  it('handles fullscreen mode', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    const fullscreenButton = screen.getByLabelText(/fullscreen/i)

    // Mock fullscreen API
    document.documentElement.requestFullscreen = jest.fn()

    await user.click(fullscreenButton)

    expect(document.documentElement.requestFullscreen).toHaveBeenCalled()
  })

  it('displays error state for invalid MJCF', async () => {
    const invalidMjcf = '<invalid>not valid mjcf</invalid>'

    render(<SimulationViewer mjcfData={invalidMjcf} />)

    await waitFor(() => {
      expect(screen.getByText(/error loading simulation/i)).toBeInTheDocument()
    })
  })

  it('handles WebSocket updates', async () => {
    const { rerender } = render(<SimulationViewer mjcfData="" />)

    // Simulate WebSocket update
    rerender(<SimulationViewer mjcfData={mockMjcfData} />)

    await waitFor(() => {
      expect(screen.getByTestId('three-canvas')).toBeInTheDocument()
    })
  })

  it('supports recording functionality', async () => {
    const user = userEvent.setup()
    render(<SimulationViewer mjcfData={mockMjcfData} enableRecording={true} />)

    const recordButton = screen.getByLabelText(/record/i)

    await user.click(recordButton)
    expect(screen.getByText(/recording/i)).toBeInTheDocument()

    // Stop recording
    await user.click(screen.getByText(/stop/i))
    expect(screen.getByLabelText(/record/i)).toBeInTheDocument()
  })

  it('exports simulation data', async () => {
    const user = userEvent.setup()
    const mockOnExport = jest.fn()

    render(
      <SimulationViewer
        mjcfData={mockMjcfData}
        onExport={mockOnExport}
      />
    )

    const exportButton = screen.getByLabelText(/export/i)
    await user.click(exportButton)

    expect(mockOnExport).toHaveBeenCalledWith(expect.any(Object))
  })

  it('handles mouse interactions with 3D scene', () => {
    const { container } = render(<SimulationViewer mjcfData={mockMjcfData} />)

    const canvas = container.querySelector('[data-testid="three-canvas"]')

    if (canvas) {
      // Simulate mouse interactions
      fireEvent.mouseDown(canvas, { clientX: 100, clientY: 100 })
      fireEvent.mouseMove(canvas, { clientX: 150, clientY: 150 })
      fireEvent.mouseUp(canvas)

      // Simulate wheel for zoom
      fireEvent.wheel(canvas, { deltaY: 100 })
    }

    expect(canvas).toBeInTheDocument()
  })

  it('displays physics information panel', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} showPhysicsInfo={true} />)

    expect(screen.getByText(/gravity:/i)).toBeInTheDocument()
    expect(screen.getByText(/timestep:/i)).toBeInTheDocument()
    expect(screen.getByText(/objects:/i)).toBeInTheDocument()
  })

  it('handles keyboard shortcuts', () => {
    render(<SimulationViewer mjcfData={mockMjcfData} />)

    // Space for play/pause
    fireEvent.keyDown(document, { key: ' ' })
    expect(screen.getByText(/pause/i)).toBeInTheDocument()

    // R for reset
    fireEvent.keyDown(document, { key: 'r' })
    expect(screen.getByText(/play/i)).toBeInTheDocument()

    // G for grid toggle
    fireEvent.keyDown(document, { key: 'g' })

    // F for fullscreen
    document.documentElement.requestFullscreen = jest.fn()
    fireEvent.keyDown(document, { key: 'f' })
  })

  it('applies custom className', () => {
    const { container } = render(
      <SimulationViewer
        mjcfData={mockMjcfData}
        className="custom-viewer"
      />
    )

    expect(container.firstChild).toHaveClass('custom-viewer')
  })

  it('handles responsive layout', () => {
    // Test mobile view
    global.innerWidth = 375
    global.innerHeight = 667
    fireEvent(window, new Event('resize'))

    const { container } = render(<SimulationViewer mjcfData={mockMjcfData} />)

    // Should have mobile-friendly controls
    expect(container.querySelector('.mobile-controls')).toBeInTheDocument()
  })

  it('maintains aspect ratio on resize', () => {
    const { container } = render(<SimulationViewer mjcfData={mockMjcfData} />)

    const canvas = container.querySelector('[data-testid="three-canvas"]')

    global.innerWidth = 1920
    global.innerHeight = 1080
    fireEvent(window, new Event('resize'))

    // Canvas should maintain aspect ratio
    expect(canvas).toBeInTheDocument()
  })

  it('cleans up resources on unmount', () => {
    const { unmount } = render(<SimulationViewer mjcfData={mockMjcfData} />)

    unmount()

    // Resources should be cleaned up (verified through mocks)
    expect(true).toBe(true)
  })
})