'use client'

/**
 * Optimized PhysicsViewer Component
 * Performance-optimized 3D physics visualization with reduced re-renders
 */

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { debounce } from 'lodash'
import {
  Play,
  Pause,
  RotateCcw,
  Maximize2,
  Settings,
  Activity,
  Zap,
  Wifi,
  WifiOff
} from 'lucide-react'
import { PhysicsClient, PhysicsFrame, ModelManifest } from '@/lib/physics-client'
import { PhysicsRenderer } from '@/lib/physics-renderer'

interface PhysicsViewerProps {
  mjcfContent?: string
  physicsSpec?: any
  autoStart?: boolean
  showControls?: boolean
  height?: string
  className?: string
}

interface SimulationStats {
  frameCount: number
  simTime: number
  fps: number
  latency: number
  connected: boolean
}

// Custom hook for animation frame updates
function useAnimationFrame(callback: (deltaTime: number) => void, deps: any[] = []) {
  const requestRef = useRef<number>()
  const previousTimeRef = useRef<number>()

  const animate = useCallback((time: number) => {
    if (previousTimeRef.current !== undefined) {
      const deltaTime = time - previousTimeRef.current
      callback(deltaTime)
    }
    previousTimeRef.current = time
    requestRef.current = requestAnimationFrame(animate)
  }, deps)

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate)
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current)
      }
    }
  }, [animate])
}

// Memoized stats display component to prevent re-renders
const StatsDisplay = React.memo(({ stats }: { stats: SimulationStats }) => {
  return (
    <div className="absolute top-2 right-2 bg-black/50 text-white p-2 rounded text-xs space-y-1">
      <div className="flex items-center gap-2">
        {stats.connected ? (
          <Wifi className="w-3 h-3 text-green-400" />
        ) : (
          <WifiOff className="w-3 h-3 text-red-400" />
        )}
        <span>{stats.connected ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div className="flex items-center gap-2">
        <Activity className="w-3 h-3" />
        <span>FPS: {stats.fps}</span>
      </div>
      <div className="flex items-center gap-2">
        <Zap className="w-3 h-3" />
        <span>Frame: {stats.frameCount}</span>
      </div>
      <div className="text-gray-400">
        Sim Time: {stats.simTime.toFixed(2)}s
      </div>
      <div className="text-gray-400">
        Latency: {stats.latency}ms
      </div>
    </div>
  )
})

StatsDisplay.displayName = 'StatsDisplay'

export function OptimizedPhysicsViewer({
  mjcfContent,
  physicsSpec,
  autoStart = false,
  showControls = true,
  height = '500px',
  className = ''
}: PhysicsViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<PhysicsRenderer | null>(null)
  const clientRef = useRef<PhysicsClient | null>(null)
  const frameQueueRef = useRef<PhysicsFrame[]>([])
  const statsRef = useRef<SimulationStats>({
    frameCount: 0,
    simTime: 0,
    fps: 0,
    latency: 0,
    connected: false
  })

  // Use refs for values that change frequently but don't need re-renders
  const lastFrameTimeRef = useRef<number>(0)
  const frameCountRef = useRef<number>(0)

  const [isConnected, setIsConnected] = useState(false)
  const [isSimulating, setIsSimulating] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [manifest, setManifest] = useState<ModelManifest | null>(null)
  const [displayStats, setDisplayStats] = useState<SimulationStats>({
    frameCount: 0,
    simTime: 0,
    fps: 0,
    latency: 0,
    connected: false
  })
  const [showSettings, setShowSettings] = useState(false)
  const [interpolation, setInterpolation] = useState(true)
  const [frameRate, setFrameRate] = useState(60)

  // Debounced stats update to reduce re-renders
  const debouncedStatsUpdate = useMemo(
    () => debounce((stats: SimulationStats) => {
      setDisplayStats(stats)
    }, 100), // Update stats max 10 times per second
    []
  )

  // Initialize physics client and renderer
  useEffect(() => {
    if (!containerRef.current) return

    // Create renderer
    const renderer = new PhysicsRenderer(containerRef.current)
    renderer.setInterpolation(interpolation)
    renderer.setFrameRate(frameRate)
    renderer.start()
    rendererRef.current = renderer

    // Create physics client
    const client = new PhysicsClient({
      url: process.env.NEXT_PUBLIC_PHYSICS_WS_URL || 'ws://localhost:8000/api/v2/physics/ws/stream',
      binaryMode: true,
      autoReconnect: true
    })
    clientRef.current = client

    // Setup optimized event handlers
    client.on('connected', () => {
      console.log('[OptimizedPhysicsViewer] Connected to server')
      setIsConnected(true)
      statsRef.current.connected = true
    })

    client.on('disconnected', () => {
      console.log('[OptimizedPhysicsViewer] Disconnected from server')
      setIsConnected(false)
      setIsSimulating(false)
      statsRef.current.connected = false
    })

    client.on('manifest', (manifest: ModelManifest) => {
      console.log('[OptimizedPhysicsViewer] Received manifest:', manifest)
      setManifest(manifest)
      renderer.initializeFromManifest(manifest)
    })

    // Queue frames instead of processing immediately
    client.on('frame', (frame: PhysicsFrame) => {
      frameQueueRef.current.push(frame)

      // Keep only last 5 frames to prevent memory buildup
      if (frameQueueRef.current.length > 5) {
        frameQueueRef.current.shift()
      }

      // Update stats refs (no re-render)
      statsRef.current.frameCount = frame.frame_id
      statsRef.current.simTime = frame.sim_time
    })

    client.on('status', (status: string) => {
      console.log('[OptimizedPhysicsViewer] Status:', status)
      if (status === 'simulation_started') {
        setIsSimulating(true)
        setIsPaused(false)
      } else if (status === 'simulation_stopped') {
        setIsSimulating(false)
      } else if (status === 'simulation_paused') {
        setIsPaused(true)
      } else if (status === 'simulation_resumed') {
        setIsPaused(false)
      }
    })

    client.on('error', (error: Error) => {
      console.error('[OptimizedPhysicsViewer] Error:', error)
    })

    // Connect to server
    client.connect()

    // Cleanup
    return () => {
      debouncedStatsUpdate.cancel()

      if (client) {
        client.disconnect()
      }

      if (renderer) {
        renderer.stop()
        renderer.dispose()
      }
    }
  }, []) // Only run once on mount

  // Process queued frames using animation frame
  useAnimationFrame((deltaTime) => {
    // Process frames from queue
    if (frameQueueRef.current.length > 0 && rendererRef.current) {
      const frame = frameQueueRef.current.shift()
      if (frame) {
        rendererRef.current.updateFrame(frame, interpolation)
      }
    }

    // Update FPS calculation
    const now = performance.now()
    if (now - lastFrameTimeRef.current >= 1000) {
      statsRef.current.fps = Math.round(frameCountRef.current * 1000 / (now - lastFrameTimeRef.current))
      frameCountRef.current = 0
      lastFrameTimeRef.current = now

      // Update display stats (debounced)
      debouncedStatsUpdate({
        ...statsRef.current,
        latency: Math.round(deltaTime)
      })
    }
    frameCountRef.current++
  }, [interpolation])

  // Load model when content changes
  useEffect(() => {
    if (!clientRef.current || !isConnected) return

    if (mjcfContent) {
      clientRef.current.loadModel(mjcfContent)
    } else if (physicsSpec) {
      clientRef.current.loadPhysicsSpec(physicsSpec)
    }
  }, [mjcfContent, physicsSpec, isConnected])

  // Auto-start simulation if requested
  useEffect(() => {
    if (autoStart && isConnected && manifest && !isSimulating) {
      handleStartSimulation()
    }
  }, [autoStart, isConnected, manifest])

  // Control handlers
  const handleStartSimulation = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.startSimulation()
    }
  }, [])

  const handleStopSimulation = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.stopSimulation()
    }
  }, [])

  const handlePauseResume = useCallback(() => {
    if (clientRef.current) {
      if (isPaused) {
        clientRef.current.resumeSimulation()
      } else {
        clientRef.current.pauseSimulation()
      }
    }
  }, [isPaused])

  const handleReset = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.resetSimulation()
      frameQueueRef.current = []
      statsRef.current.frameCount = 0
      statsRef.current.simTime = 0
    }
  }, [])

  const handleFullscreen = useCallback(() => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen()
      } else {
        containerRef.current.requestFullscreen()
      }
    }
  }, [])

  // Settings handlers with memoization
  const handleInterpolationChange = useCallback((enabled: boolean) => {
    setInterpolation(enabled)
    if (rendererRef.current) {
      rendererRef.current.setInterpolation(enabled)
    }
  }, [])

  const handleFrameRateChange = useCallback((rate: number) => {
    setFrameRate(rate)
    if (rendererRef.current) {
      rendererRef.current.setFrameRate(rate)
    }
  }, [])

  return (
    <div className={`relative ${className}`} style={{ height }}>
      {/* 3D Viewer Container */}
      <div
        ref={containerRef}
        className="w-full h-full bg-gradient-to-b from-gray-900 to-black rounded-lg overflow-hidden"
      />

      {/* Stats Display (Memoized) */}
      <StatsDisplay stats={displayStats} />

      {/* Controls */}
      {showControls && (
        <div className="absolute bottom-4 left-4 flex gap-2">
          {!isSimulating ? (
            <button
              onClick={handleStartSimulation}
              disabled={!isConnected || !manifest}
              className="p-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-600 text-white rounded-lg transition-colors"
              title="Start Simulation"
            >
              <Play className="w-5 h-5" />
            </button>
          ) : (
            <>
              <button
                onClick={handlePauseResume}
                className="p-2 bg-yellow-500 hover:bg-yellow-600 text-white rounded-lg transition-colors"
                title={isPaused ? "Resume" : "Pause"}
              >
                {isPaused ? <Play className="w-5 h-5" /> : <Pause className="w-5 h-5" />}
              </button>
              <button
                onClick={handleStopSimulation}
                className="p-2 bg-red-500 hover:bg-red-600 text-white rounded-lg transition-colors"
                title="Stop Simulation"
              >
                <Pause className="w-5 h-5" />
              </button>
            </>
          )}

          <button
            onClick={handleReset}
            disabled={!isConnected}
            className="p-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white rounded-lg transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-5 h-5" />
          </button>

          <button
            onClick={handleFullscreen}
            className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-5 h-5" />
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="absolute top-2 left-2 bg-black/80 text-white p-4 rounded-lg space-y-3">
          <h3 className="font-semibold">Rendering Settings</h3>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="interpolation"
              checked={interpolation}
              onChange={(e) => handleInterpolationChange(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="interpolation" className="text-sm">
              Frame Interpolation
            </label>
          </div>

          <div className="space-y-1">
            <label className="text-sm">Target FPS: {frameRate}</label>
            <input
              type="range"
              min="30"
              max="120"
              step="30"
              value={frameRate}
              onChange={(e) => handleFrameRateChange(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>
      )}
    </div>
  )
}