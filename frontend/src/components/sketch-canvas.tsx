"use client"

import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Button } from './ui/button'
import { Trash2, Download, Upload } from 'lucide-react'
import { motion } from 'framer-motion'

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
      }
    }
  }, [isDrawing, onSketchChange])

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
      
      {/* Drawing hint */}
      <div className="absolute bottom-4 left-4 text-sm text-gray-500 bg-white/90 px-3 py-1 rounded-lg backdrop-blur-sm">
        Draw your physics idea here
      </div>
    </motion.div>
  )
}