/**
 * Hook for managing creation mode state
 */

import { useState, useEffect, useCallback } from 'react'

export type CreationMode = 'physics' | 'games' | 'vr'

interface ModeState {
  currentMode: CreationMode
  setMode: (mode: CreationMode) => void
  modeHistory: CreationMode[]
  canSwitchMode: boolean
  switchMode: (newMode: CreationMode) => void
}

export function useMode(initialMode: CreationMode = 'physics'): ModeState {
  const [currentMode, setCurrentMode] = useState<CreationMode>(initialMode)
  const [modeHistory, setModeHistory] = useState<CreationMode[]>([initialMode])
  const [hasUnsavedWork, setHasUnsavedWork] = useState(false)

  // Load mode from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('virtualforge_mode')
    if (saved && ['physics', 'games', 'vr'].includes(saved)) {
      setCurrentMode(saved as CreationMode)
    }
  }, [])

  // Save mode to localStorage
  const setMode = useCallback((mode: CreationMode) => {
    setCurrentMode(mode)
    localStorage.setItem('virtualforge_mode', mode)
    setModeHistory(prev => [...prev, mode])
  }, [])

  // Switch mode with optional confirmation
  const switchMode = useCallback((newMode: CreationMode) => {
    if (hasUnsavedWork) {
      const confirm = window.confirm(
        'You have unsaved work. Switching modes will clear it. Continue?'
      )
      if (!confirm) return
    }

    setMode(newMode)
  }, [hasUnsavedWork, setMode])

  return {
    currentMode,
    setMode,
    modeHistory,
    canSwitchMode: !hasUnsavedWork,
    switchMode
  }
}