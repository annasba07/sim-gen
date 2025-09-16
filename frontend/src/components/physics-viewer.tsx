'use client'

/**
 * PhysicsViewer Component
 * Real-time 3D physics visualization using binary WebSocket streaming
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
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

export function PhysicsViewer({
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
  const statsIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const [isConnected, setIsConnected] = useState(false)
  const [isSimulating, setIsSimulating] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [manifest, setManifest] = useState<ModelManifest | null>(null)
  const [stats, setStats] = useState<SimulationStats>({
    frameCount: 0,
    simTime: 0,
    fps: 0,
    latency: 0,
    connected: false
  })
  const [showSettings, setShowSettings] = useState(false)
  const [interpolation, setInterpolation] = useState(true)
  const [frameRate, setFrameRate] = useState(60)

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

    // Setup event handlers
    client.on('connected', () => {
      console.log('[PhysicsViewer] Connected to server')
      setIsConnected(true)
      setStats(prev => ({ ...prev, connected: true }))
    })

    client.on('disconnected', () => {
      console.log('[PhysicsViewer] Disconnected from server')
      setIsConnected(false)
      setIsSimulating(false)
      setStats(prev => ({ ...prev, connected: false }))
    })

    client.on('manifest', (manifest: ModelManifest) => {
      console.log('[PhysicsViewer] Received manifest:', manifest)
      setManifest(manifest)
      renderer.initializeFromManifest(manifest)
    })

    client.on('frame', (frame: PhysicsFrame) => {
      renderer.updateFrame(frame, interpolation)
      setStats(prev => ({
        ...prev,
        frameCount: frame.frame_id,
        simTime: frame.sim_time
      }))
    })

    client.on('status', (status: string) => {
      console.log('[PhysicsViewer] Status:', status)
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
      console.error('[PhysicsViewer] Error:', error)
    })

    // Connect to server
    client.connect()

    // Setup FPS counter
    let lastTime = performance.now()
    let frameCount = 0

    statsIntervalRef.current = setInterval(() => {
      const now = performance.now()
      const elapsed = (now - lastTime) / 1000
      const fps = frameCount / elapsed

      setStats(prev => ({
        ...prev,
        fps: Math.round(fps),
        latency: Math.round(now - lastTime)
      }))

      frameCount = 0
      lastTime = now
    }, 1000)

    // Track frames for FPS
    client.on('frame', () => {
      frameCount++
    })

    // Cleanup
    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current)
      }
      client.disconnect()
      renderer.dispose()
    }
  }, [])

  // Update interpolation settings
  useEffect(() => {
    if (rendererRef.current) {
      rendererRef.current.setInterpolation(interpolation)
    }
  }, [interpolation])

  // Update frame rate
  useEffect(() => {
    if (rendererRef.current) {
      rendererRef.current.setFrameRate(frameRate)
    }
  }, [frameRate])

  // Load model when content changes
  useEffect(() => {
    if (mjcfContent && clientRef.current && isConnected) {
      console.log('[PhysicsViewer] Loading MJCF model')
      clientRef.current.loadModel(mjcfContent)

      if (autoStart) {
        setTimeout(() => {
          startSimulation()
        }, 500)
      }
    }
  }, [mjcfContent, isConnected, autoStart])

  // Compile PhysicsSpec if provided
  useEffect(() => {
    if (physicsSpec && isConnected) {
      compilePhysicsSpec(physicsSpec)
    }
  }, [physicsSpec, isConnected])

  const compilePhysicsSpec = async (spec: any) => {
    try {
      const response = await fetch('/api/v2/physics/compile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec })
      })

      const data = await response.json()
      if (data.success && data.mjcf_xml) {
        if (clientRef.current) {
          clientRef.current.loadModel(data.mjcf_xml)
        }
      }
    } catch (error) {
      console.error('[PhysicsViewer] Failed to compile PhysicsSpec:', error)
    }
  }

  const startSimulation = useCallback(() => {
    if (clientRef.current && isConnected) {
      clientRef.current.startSimulation()
      setIsSimulating(true)
      setIsPaused(false)
    }
  }, [isConnected])

  const stopSimulation = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.stopSimulation()
      setIsSimulating(false)
      setIsPaused(false)
    }
  }, [])

  const pauseSimulation = useCallback(() => {
    if (clientRef.current && isSimulating) {
      if (isPaused) {
        clientRef.current.resumeSimulation()
        setIsPaused(false)
      } else {
        clientRef.current.pauseSimulation()
        setIsPaused(true)
      }
    }
  }, [isSimulating, isPaused])

  const resetSimulation = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.resetSimulation()
      if (rendererRef.current) {
        rendererRef.current.resetCamera()
      }
    }
  }, [])

  const toggleFullscreen = useCallback(() => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen()
      } else {
        containerRef.current.requestFullscreen()
      }
    }
  }, [])

  return (
    <div className={`relative bg-gray-50 rounded-xl overflow-hidden ${className}`}>
      {/* 3D Viewport */}
      <div
        ref={containerRef}
        style={{ height }}
        className="w-full"
      />

      {/* Controls Overlay */}
      {showControls && (
        <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none">
          {/* Left Controls */}
          <div className="flex flex-col gap-2 pointer-events-auto">
            {/* Connection Status */}
            <div className={`
              px-3 py-1.5 rounded-lg backdrop-blur-sm flex items-center gap-2 text-sm
              ${isConnected
                ? 'bg-green-500/20 text-green-700 border border-green-500/30'
                : 'bg-red-500/20 text-red-700 border border-red-500/30'}
            `}>
              {isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>

            {/* Playback Controls */}
            {isConnected && (
              <div className="flex gap-1 bg-white/90 backdrop-blur-sm rounded-lg p-1 border border-gray-200">
                {!isSimulating ? (
                  <button
                    onClick={startSimulation}
                    className="p-2 hover:bg-gray-100 rounded-md transition-colors"
                    title="Start Simulation"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                ) : (
                  <>
                    <button
                      onClick={pauseSimulation}
                      className="p-2 hover:bg-gray-100 rounded-md transition-colors"
                      title={isPaused ? 'Resume' : 'Pause'}
                    >
                      {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={stopSimulation}
                      className="p-2 hover:bg-gray-100 rounded-md transition-colors"
                      title="Stop"
                    >
                      <div className="w-4 h-4 bg-current rounded-sm" />
                    </button>
                  </>
                )}
                <button
                  onClick={resetSimulation}
                  className="p-2 hover:bg-gray-100 rounded-md transition-colors"
                  title="Reset"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={toggleFullscreen}
                  className="p-2 hover:bg-gray-100 rounded-md transition-colors"
                  title="Fullscreen"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className={`p-2 hover:bg-gray-100 rounded-md transition-colors ${
                    showSettings ? 'bg-gray-100' : ''
                  }`}
                  title="Settings"
                >
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Settings Panel */}
            {showSettings && (
              <div className="bg-white/90 backdrop-blur-sm rounded-lg p-3 border border-gray-200 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Interpolation</span>
                  <input
                    type="checkbox"
                    checked={interpolation}
                    onChange={(e) => setInterpolation(e.target.checked)}
                    className="rounded"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Target FPS</span>
                  <select
                    value={frameRate}
                    onChange={(e) => setFrameRate(Number(e.target.value))}
                    className="text-sm rounded border-gray-300"
                  >
                    <option value={30}>30</option>
                    <option value={60}>60</option>
                    <option value={120}>120</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* Right Stats */}
          <div className="pointer-events-auto">
            <div className="bg-black/80 backdrop-blur-sm rounded-lg p-3 text-white space-y-1 font-mono text-xs">
              {manifest && (
                <>
                  <div className="text-green-400 font-bold">{manifest.model_name}</div>
                  <div className="flex items-center gap-1">
                    <Zap className="w-3 h-3" />
                    Bodies: {manifest.nbody}
                  </div>
                  <div>DOFs: {manifest.nq}</div>
                  {manifest.nu > 0 && <div>Actuators: {manifest.nu}</div>}
                </>
              )}
              {isSimulating && (
                <>
                  <div className="border-t border-white/20 pt-1 mt-1">
                    <div className="flex items-center gap-1">
                      <Activity className="w-3 h-3" />
                      FPS: {stats.fps}
                    </div>
                    <div>Frame: {stats.frameCount}</div>
                    <div>Time: {stats.simTime.toFixed(2)}s</div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}
      {isSimulating && (
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/20 to-transparent p-4">
          <div className="flex justify-between items-center text-white text-sm">
            <div className="flex items-center gap-3">
              {isPaused && (
                <span className="px-2 py-0.5 bg-yellow-500/80 rounded text-xs font-medium">
                  PAUSED
                </span>
              )}
              <span className="font-mono">
                {stats.simTime.toFixed(3)}s
              </span>
            </div>
            <div className="flex items-center gap-3 font-mono text-xs">
              <span>{stats.fps} FPS</span>
              <span className="text-green-400">Binary Mode</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}