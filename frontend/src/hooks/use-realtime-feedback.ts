/**
 * Hook for real-time sketch feedback via WebSocket
 */

import { useEffect, useState, useCallback, useRef } from 'react'
import { io, Socket } from 'socket.io-client'

export interface FeedbackMessage {
  type: 'live_feedback' | 'error' | 'suggestion'
  message: string
  confidence?: number
  suggestions?: string[]
  physics_hints?: string[]
  visual_overlays?: Array<{
    type: string
    bbox: number[]
    confidence: number
    color: string
    label: string
  }>
}

interface UseRealtimeFeedbackOptions {
  enabled?: boolean
  debounceMs?: number
  wsUrl?: string
}

export function useRealtimeFeedback(options: UseRealtimeFeedbackOptions = {}) {
  const {
    enabled = process.env.NEXT_PUBLIC_ENABLE_REALTIME === 'true',
    debounceMs = 500,
    wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  } = options

  const [socket, setSocket] = useState<Socket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [feedback, setFeedback] = useState<FeedbackMessage | null>(null)
  const [sessionId] = useState(() => `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`)
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null)

  // Initialize WebSocket connection
  useEffect(() => {
    if (!enabled) return

    const socketInstance = io(wsUrl, {
      path: '/realtime/sketch-feedback',
      query: { sessionId },
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      transports: ['websocket']
    })

    socketInstance.on('connect', () => {
      console.log('Real-time feedback connected')
      setIsConnected(true)
    })

    socketInstance.on('disconnect', () => {
      console.log('Real-time feedback disconnected')
      setIsConnected(false)
    })

    socketInstance.on('feedback', (data: FeedbackMessage) => {
      setFeedback(data)
    })

    socketInstance.on('error', (error: any) => {
      console.error('WebSocket error:', error)
      setFeedback({
        type: 'error',
        message: 'Connection error. Feedback temporarily unavailable.'
      })
    })

    setSocket(socketInstance)

    return () => {
      socketInstance.disconnect()
    }
  }, [enabled, wsUrl, sessionId])

  // Send sketch data for analysis (debounced)
  const sendSketchData = useCallback((sketchData: string) => {
    if (!socket || !isConnected) return

    // Clear existing debounce timer
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current)
    }

    // Set new debounce timer
    debounceTimerRef.current = setTimeout(() => {
      socket.emit('analyze_sketch', {
        session_id: sessionId,
        sketch_data: sketchData,
        timestamp: Date.now()
      })
    }, debounceMs)
  }, [socket, isConnected, sessionId, debounceMs])

  // Clear feedback
  const clearFeedback = useCallback(() => {
    setFeedback(null)
  }, [])

  // Cleanup
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current)
      }
    }
  }, [])

  return {
    isConnected,
    feedback,
    sendSketchData,
    clearFeedback,
    sessionId
  }
}