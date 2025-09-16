'use client'

/**
 * ActuatorControls Component
 * Real-time control interface for physics simulation actuators
 */

import React, { useState, useCallback, useEffect } from 'react'
import { Sliders, Gamepad2, RotateCw, Zap } from 'lucide-react'

export interface ActuatorInfo {
  id: string
  name: string
  type: 'motor' | 'position' | 'velocity'
  ctrlrange?: [number, number]
  forcerange?: [number, number]
  currentValue: number
}

interface ActuatorControlsProps {
  actuators: ActuatorInfo[]
  onControlChange: (values: Float32Array) => void
  className?: string
  compact?: boolean
}

export function ActuatorControls({
  actuators,
  onControlChange,
  className = '',
  compact = false
}: ActuatorControlsProps) {
  const [controlValues, setControlValues] = useState<number[]>(() =>
    actuators.map(() => 0)
  )
  const [isJoystickMode, setIsJoystickMode] = useState(false)
  const [gamepadIndex, setGamepadIndex] = useState<number | null>(null)

  // Handle slider change
  const handleSliderChange = useCallback(
    (index: number, value: number) => {
      const newValues = [...controlValues]
      newValues[index] = value
      setControlValues(newValues)

      // Send to physics simulation
      onControlChange(new Float32Array(newValues))
    },
    [controlValues, onControlChange]
  )

  // Reset all controls to zero
  const resetControls = useCallback(() => {
    const zeros = new Array(actuators.length).fill(0)
    setControlValues(zeros)
    onControlChange(new Float32Array(zeros))
  }, [actuators.length, onControlChange])

  // Apply preset patterns
  const applyPreset = useCallback(
    (preset: string) => {
      let newValues: number[] = []

      switch (preset) {
        case 'sine':
          // Sine wave pattern
          newValues = actuators.map((_, i) =>
            Math.sin((Date.now() / 1000 + i) * 2)
          )
          break
        case 'step':
          // Step function
          newValues = actuators.map((_, i) =>
            i % 2 === 0 ? 1 : -1
          )
          break
        case 'ramp':
          // Linear ramp
          newValues = actuators.map((_, i) =>
            (i / (actuators.length - 1)) * 2 - 1
          )
          break
        default:
          newValues = new Array(actuators.length).fill(0)
      }

      // Clamp to control ranges
      newValues = newValues.map((v, i) => {
        const actuator = actuators[i]
        if (actuator.ctrlrange) {
          return Math.max(
            actuator.ctrlrange[0],
            Math.min(actuator.ctrlrange[1], v)
          )
        }
        return v
      })

      setControlValues(newValues)
      onControlChange(new Float32Array(newValues))
    },
    [actuators, onControlChange]
  )

  // Gamepad support
  useEffect(() => {
    if (!isJoystickMode) return

    const updateGamepad = () => {
      const gamepads = navigator.getGamepads()
      const gamepad = gamepads[gamepadIndex ?? 0]

      if (gamepad && gamepad.connected) {
        const newValues = [...controlValues]

        // Map gamepad axes to actuators
        actuators.forEach((actuator, i) => {
          if (i < gamepad.axes.length) {
            let value = gamepad.axes[i]

            // Apply deadzone
            if (Math.abs(value) < 0.1) {
              value = 0
            }

            // Map to control range
            if (actuator.ctrlrange) {
              const [min, max] = actuator.ctrlrange
              value = ((value + 1) / 2) * (max - min) + min
            }

            newValues[i] = value
          }
        })

        setControlValues(newValues)
        onControlChange(new Float32Array(newValues))
      }

      if (isJoystickMode) {
        requestAnimationFrame(updateGamepad)
      }
    }

    const handleGamepadConnected = (e: GamepadEvent) => {
      console.log('Gamepad connected:', e.gamepad)
      setGamepadIndex(e.gamepad.index)
    }

    const handleGamepadDisconnected = () => {
      console.log('Gamepad disconnected')
      setGamepadIndex(null)
    }

    window.addEventListener('gamepadconnected', handleGamepadConnected)
    window.addEventListener('gamepaddisconnected', handleGamepadDisconnected)

    if (isJoystickMode) {
      updateGamepad()
    }

    return () => {
      window.removeEventListener('gamepadconnected', handleGamepadConnected)
      window.removeEventListener('gamepaddisconnected', handleGamepadDisconnected)
    }
  }, [isJoystickMode, gamepadIndex, actuators, controlValues, onControlChange])

  if (actuators.length === 0) {
    return (
      <div className={`bg-gray-50 rounded-lg p-4 text-center text-gray-500 ${className}`}>
        <Zap className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No actuators in this model</p>
      </div>
    )
  }

  if (compact) {
    // Compact horizontal layout
    return (
      <div className={`bg-white rounded-lg border border-gray-200 p-3 ${className}`}>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Sliders className="w-4 h-4" />
            <span>Controls</span>
          </div>
          <div className="flex-1 flex gap-3">
            {actuators.map((actuator, index) => (
              <div key={actuator.id} className="flex-1 flex items-center gap-2">
                <span className="text-xs text-gray-600">{actuator.name}</span>
                <input
                  type="range"
                  min={actuator.ctrlrange?.[0] ?? -1}
                  max={actuator.ctrlrange?.[1] ?? 1}
                  step="0.01"
                  value={controlValues[index]}
                  onChange={(e) => handleSliderChange(index, parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="text-xs font-mono w-12 text-right">
                  {controlValues[index].toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <button
            onClick={resetControls}
            className="p-1.5 hover:bg-gray-100 rounded-md transition-colors"
            title="Reset"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>
    )
  }

  // Full layout
  return (
    <div className={`bg-white rounded-xl border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-gray-600" />
          <h3 className="font-semibold">Actuator Controls</h3>
          <span className="text-sm text-gray-500">
            ({actuators.length} actuators)
          </span>
        </div>
        <div className="flex items-center gap-2">
          {/* Presets */}
          <select
            onChange={(e) => applyPreset(e.target.value)}
            className="text-sm rounded-md border-gray-300"
            defaultValue=""
          >
            <option value="" disabled>
              Presets
            </option>
            <option value="sine">Sine Wave</option>
            <option value="step">Step</option>
            <option value="ramp">Ramp</option>
          </select>

          {/* Gamepad toggle */}
          <button
            onClick={() => setIsJoystickMode(!isJoystickMode)}
            className={`p-2 rounded-md transition-colors ${
              isJoystickMode
                ? 'bg-blue-500 text-white'
                : 'hover:bg-gray-100'
            }`}
            title="Gamepad Control"
          >
            <Gamepad2 className="w-4 h-4" />
          </button>

          {/* Reset button */}
          <button
            onClick={resetControls}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title="Reset All"
          >
            <RotateCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Gamepad status */}
      {isJoystickMode && (
        <div className={`px-4 py-2 text-sm ${
          gamepadIndex !== null
            ? 'bg-green-50 text-green-700'
            : 'bg-yellow-50 text-yellow-700'
        }`}>
          {gamepadIndex !== null
            ? `Gamepad connected (index: ${gamepadIndex})`
            : 'Waiting for gamepad... Press any button on your controller'}
        </div>
      )}

      {/* Actuator sliders */}
      <div className="p-4 space-y-4">
        {actuators.map((actuator, index) => (
          <div key={actuator.id} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{actuator.name}</span>
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                  {actuator.type}
                </span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-gray-500">
                  {actuator.ctrlrange?.[0] ?? -1}
                </span>
                <span className="font-mono font-medium w-16 text-center">
                  {controlValues[index].toFixed(2)}
                </span>
                <span className="text-gray-500">
                  {actuator.ctrlrange?.[1] ?? 1}
                </span>
              </div>
            </div>
            <div className="relative">
              <input
                type="range"
                min={actuator.ctrlrange?.[0] ?? -1}
                max={actuator.ctrlrange?.[1] ?? 1}
                step="0.01"
                value={controlValues[index]}
                onChange={(e) => handleSliderChange(index, parseFloat(e.target.value))}
                className="w-full"
                style={{
                  background: `linear-gradient(to right,
                    #3b82f6 0%,
                    #3b82f6 ${((controlValues[index] - (actuator.ctrlrange?.[0] ?? -1)) /
                      ((actuator.ctrlrange?.[1] ?? 1) - (actuator.ctrlrange?.[0] ?? -1))) * 100}%,
                    #e5e7eb ${((controlValues[index] - (actuator.ctrlrange?.[0] ?? -1)) /
                      ((actuator.ctrlrange?.[1] ?? 1) - (actuator.ctrlrange?.[0] ?? -1))) * 100}%,
                    #e5e7eb 100%)`
                }}
              />
              {/* Zero line indicator */}
              {(actuator.ctrlrange?.[0] ?? -1) < 0 &&
               (actuator.ctrlrange?.[1] ?? 1) > 0 && (
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-0.5 h-4 bg-gray-400 pointer-events-none"
                  style={{
                    left: `${(0 - (actuator.ctrlrange?.[0] ?? -1)) /
                      ((actuator.ctrlrange?.[1] ?? 1) - (actuator.ctrlrange?.[0] ?? -1)) * 100}%`
                  }}
                />
              )}
            </div>
            {/* Force range indicator */}
            {actuator.forcerange && (
              <div className="text-xs text-gray-500">
                Force range: [{actuator.forcerange[0]}, {actuator.forcerange[1]}] N
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}