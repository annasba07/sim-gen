"use client"

import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Button } from './ui/button'
import { Trash2, Download, Upload, Info, CheckCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRealtimeFeedback } from '@/hooks/use-realtime-feedback'

interface SketchCanvasProps {
  width?: number
  height?: number
  onSketchChange?: (dataURL: string) => void
  className?: string
}

interface Point {
  x: number
  y: number
}

export function SketchCanvas({
  width = 600,
  height = 400,
  onSketchChange,
  className = ""
}: SketchCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [lastPoint, setLastPoint] = useState<Point | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Real-time feedback integration
  const { isConnected, feedback, sendSketchData } = useRealtimeFeedback()

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = width
    canvas.height = height

    // Set drawing style
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, width, height)
    ctx.strokeStyle = '#000000'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }, [width, height])

  // Get point coordinates relative to canvas
  const getPoint = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>): Point => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }

    const rect = canvas.getBoundingClientRect()
    
    if ('touches' in e) {
      // Touch event
      const touch = e.touches[0] || e.changedTouches[0]
      return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
      }
    } else {
      // Mouse event
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      }
    }
  }, [])

  // Start drawing
  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const point = getPoint(e)
    setIsDrawing(true)
    setLastPoint(point)
  }, [getPoint])

  // Continue drawing
  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    if (!isDrawing || !lastPoint) return

    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const currentPoint = getPoint(e)
    
    ctx.beginPath()
    ctx.moveTo(lastPoint.x, lastPoint.y)
    ctx.lineTo(currentPoint.x, currentPoint.y)
    ctx.stroke()
    
    setLastPoint(currentPoint)
  }, [isDrawing, lastPoint, getPoint])

  // Stop drawing
  const stopDrawing = useCallback(() => {
    if (isDrawing) {
      setIsDrawing(false)
      setLastPoint(null)

      // Notify parent of canvas change
      const canvas = canvasRef.current
      if (canvas && onSketchChange) {
        const dataURL = canvas.toDataURL('image/png')
        onSketchChange(dataURL)

        // Send to real-time feedback if connected
        if (isConnected) {
          sendSketchData(dataURL)
        }
      }
    }
  }, [isDrawing, onSketchChange, isConnected, sendSketchData])

  // Clear canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    if (onSketchChange) {
      const dataURL = canvas.toDataURL('image/png')
      onSketchChange(dataURL)
    }
  }, [onSketchChange])

  // Download sketch
  const downloadSketch = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const link = document.createElement('a')
    link.download = 'physics-sketch.png'
    link.href = canvas.toDataURL('image/png')
    link.click()
  }, [])

  // Upload image
  const uploadImage = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const reader = new FileReader()
    reader.onload = (event) => {
      const img = new Image()
      img.onload = () => {
        const canvas = canvasRef.current
        const ctx = canvas?.getContext('2d')
        if (!canvas || !ctx) return

        // Clear canvas and draw image
        ctx.fillStyle = 'white'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        // Scale image to fit canvas while maintaining aspect ratio
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height)
        const x = (canvas.width - img.width * scale) / 2
        const y = (canvas.height - img.height * scale) / 2
        
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale)
        
        if (onSketchChange) {
          const dataURL = canvas.toDataURL('image/png')
          onSketchChange(dataURL)
        }
      }
      img.src = event.target?.result as string
    }
    reader.readAsDataURL(file)
  }, [onSketchChange])

  return (
    <motion.div 
      className={`relative bg-white rounded-xl shadow-2xl border-2 border-gray-200 overflow-hidden ${className}`}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Canvas */}
      <canvas
        ref={canvasRef}
        className="block cursor-crosshair touch-none"
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={stopDrawing}
        style={{ width: '100%', height: 'auto' }}
      />
      
      {/* Controls */}
      <div className="absolute top-4 right-4 flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => fileInputRef.current?.click()}
          className="bg-white/90 backdrop-blur-sm hover:bg-white"
        >
          <Upload className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={downloadSketch}
          className="bg-white/90 backdrop-blur-sm hover:bg-white"
        >
          <Download className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={clearCanvas}
          className="bg-white/90 backdrop-blur-sm hover:bg-white text-red-600 hover:text-red-700"
        >
          <Trash2 className="w-4 h-4" />
        </Button>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={uploadImage}
        className="hidden"
      />
      
      {/* Drawing hint / Real-time feedback */}
      <AnimatePresence mode="wait">
        {feedback ? (
          <motion.div
            key="feedback"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className={`absolute bottom-4 left-4 right-4 p-3 rounded-lg backdrop-blur-sm ${
              feedback.type === 'error'
                ? 'bg-red-100/90 border border-red-300'
                : feedback.confidence && feedback.confidence > 0.7
                ? 'bg-green-100/90 border border-green-300'
                : 'bg-blue-100/90 border border-blue-300'
            }`}
          >
            <div className="flex items-start gap-2">
              {feedback.type === 'error' ? (
                <Info className="w-4 h-4 text-red-600 mt-0.5" />
              ) : feedback.confidence && feedback.confidence > 0.7 ? (
                <CheckCircle className="w-4 h-4 text-green-600 mt-0.5" />
              ) : (
                <Info className="w-4 h-4 text-blue-600 mt-0.5" />
              )}
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-800">
                  {feedback.message}
                </p>
                {feedback.suggestions && feedback.suggestions.length > 0 && (
                  <ul className="mt-1 text-xs text-gray-600">
                    {feedback.suggestions.map((suggestion, idx) => (
                      <li key={idx}>• {suggestion}</li>
                    ))}
                  </ul>
                )}
                {feedback.physics_hints && feedback.physics_hints.length > 0 && (
                  <p className="mt-1 text-xs text-gray-600 italic">
                    {feedback.physics_hints[0]}
                  </p>
                )}
              </div>
              {feedback.confidence && (
                <div className="text-xs text-gray-500">
                  {Math.round(feedback.confidence * 100)}%
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="hint"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-4 left-4 text-sm text-gray-500 bg-white/90 px-3 py-1 rounded-lg backdrop-blur-sm"
          >
            <div className="flex items-center gap-2">
              {isConnected && (
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
              )}
              <span>Draw your physics idea here</span>
              {isConnected && (
                <span className="text-xs text-gray-400">(Live feedback enabled)</span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}