"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ModeSelector } from '@/components/shared/mode-selector'
import { SketchCanvas } from '@/components/sketch-canvas'
import { SimulationViewer } from '@/components/simulation-viewer'
import { Button } from '@/components/ui/button'
import { useMode } from '@/hooks/use-mode'
import {
  ArrowLeft,
  Sparkles,
  Wand2,
  Play,
  Download,
  Share2,
  Settings
} from 'lucide-react'

export default function VirtualForgePage() {
  const { currentMode, setMode } = useMode()
  const [showModeSelector, setShowModeSelector] = useState(true)
  const [prompt, setPrompt] = useState('')
  const [sketchData, setSketchData] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleModeSelect = (modeId: string) => {
    setMode(modeId as any)
    setShowModeSelector(false)
  }

  const handleBackToModeSelector = () => {
    setShowModeSelector(true)
    setResult(null)
  }

  const handleGenerate = async () => {
    setIsGenerating(true)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v2/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: currentMode,
          prompt,
          sketch_data: sketchData ? sketchData.split(',')[1] : null
        })
      })

      const data = await response.json()

      if (data.success) {
        setResult(data)
      } else {
        alert(`Generation failed: ${data.errors?.join(', ')}`)
      }
    } catch (error) {
      console.error('Generation error:', error)
      alert('Failed to generate. Please try again.')
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      {/* Header */}
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {!showModeSelector && (
              <button
                onClick={handleBackToModeSelector}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
            )}

            <div className="flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-purple-600" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                VirtualForge
              </h1>
            </div>

            {!showModeSelector && (
              <div className="flex items-center gap-2 ml-4 px-3 py-1 bg-purple-100 rounded-full">
                <span className="text-2xl">{getModeIcon(currentMode)}</span>
                <span className="text-sm font-semibold text-purple-700">
                  {getModeName(currentMode)}
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm">
              My Projects
            </Button>
            <Button variant="ghost" size="sm">
              <Settings className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          {showModeSelector ? (
            <motion.div
              key="mode-selector"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <ModeSelector
                onSelectMode={handleModeSelect}
                selectedMode={currentMode}
              />
            </motion.div>
          ) : (
            <motion.div
              key="creator"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-8"
            >
              {/* Creation Interface */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Input Section */}
                <div className="space-y-6">
                  {/* Prompt Input */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-700">
                      Describe What You Want to Create
                    </label>
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder={getPlaceholder(currentMode)}
                      className="w-full h-32 p-4 border-2 border-gray-200 rounded-xl focus:ring-2
                        focus:ring-purple-500 focus:border-transparent resize-none text-gray-700"
                    />
                  </div>

                  {/* Sketch Canvas */}
                  <div className="space-y-3">
                    <label className="text-sm font-semibold text-gray-700">
                      Sketch Your Idea (Optional)
                    </label>
                    <SketchCanvas
                      width={500}
                      height={300}
                      onSketchChange={setSketchData}
                      className="w-full"
                    />
                  </div>

                  {/* Generate Button */}
                  <Button
                    onClick={handleGenerate}
                    disabled={isGenerating || !prompt}
                    size="lg"
                    className="w-full bg-gradient-to-r from-purple-500 to-blue-600 hover:from-purple-600
                      hover:to-blue-700 text-white font-semibold"
                  >
                    {isGenerating ? (
                      <div className="flex items-center gap-3">
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Creating...</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-3">
                        <Wand2 className="w-5 h-5" />
                        <span>Generate {getModeName(currentMode)}</span>
                      </div>
                    )}
                  </Button>
                </div>

                {/* Output Section */}
                <div className="space-y-6">
                  <label className="text-sm font-semibold text-gray-700">
                    Preview
                  </label>

                  {result ? (
                    <ResultDisplay result={result} mode={currentMode} />
                  ) : (
                    <div className="bg-white rounded-xl border-2 border-dashed border-gray-300 h-96 flex items-center justify-center">
                      <div className="text-center space-y-3">
                        <div className="text-6xl">{getModeIcon(currentMode)}</div>
                        <p className="text-gray-500">
                          Your {getModeName(currentMode).toLowerCase()} will appear here
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Mode-specific tips */}
              <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
                <h3 className="font-semibold text-blue-900 mb-2">💡 Tips for {getModeName(currentMode)}</h3>
                <ul className="text-sm text-blue-800 space-y-1">
                  {getModeTips(currentMode).map((tip, i) => (
                    <li key={i}>• {tip}</li>
                  ))}
                </ul>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}

function ResultDisplay({ result, mode }: { result: any; mode: string }) {
  if (mode === 'physics') {
    return (
      <div className="space-y-4">
        <SimulationViewer mjcfContent={result.output.mjcf_xml} />
        <div className="flex gap-2">
          <Button size="sm" className="flex-1">
            <Play className="w-4 h-4 mr-2" />
            Run Simulation
          </Button>
          <Button size="sm" variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button size="sm" variant="outline">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
        </div>
      </div>
    )
  }

  if (mode === 'games') {
    return (
      <div className="space-y-4">
        <div className="bg-white rounded-xl border-2 border-gray-200 h-96 flex items-center justify-center">
          <p className="text-gray-500">Game preview coming soon</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" className="flex-1">
            <Play className="w-4 h-4 mr-2" />
            Play Game
          </Button>
          <Button size="sm" variant="outline">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <Button size="sm" variant="outline">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
        </div>
      </div>
    )
  }

  return <div>Result preview for {mode}</div>
}

function getModeIcon(mode: string): string {
  const icons: Record<string, string> = {
    physics: '🔬',
    games: '🎮',
    vr: '🌐'
  }
  return icons[mode] || '✨'
}

function getModeName(mode: string): string {
  const names: Record<string, string> = {
    physics: 'Physics Simulation',
    games: 'Game',
    vr: 'VR Experience'
  }
  return names[mode] || 'Creation'
}

function getPlaceholder(mode: string): string {
  const placeholders: Record<string, string> = {
    physics: 'Example: Create a pendulum that swings back and forth with adjustable gravity...',
    games: 'Example: Make a platformer where a cat collects stars while avoiding robots...',
    vr: 'Example: Build a virtual art gallery where users can walk around and view paintings...'
  }
  return placeholders[mode] || 'Describe what you want to create...'
}

function getModeTips(mode: string): string[] {
  const tips: Record<string, string[]> = {
    physics: [
      'Be specific about object properties (mass, size, material)',
      'Describe the physics behavior you want to see',
      'Sketching helps AI understand spatial relationships'
    ],
    games: [
      'Describe the player character and controls',
      'Mention enemies, obstacles, and collectibles',
      'Specify the win/lose conditions'
    ],
    vr: [
      'Describe the environment and atmosphere',
      'Mention interactive elements',
      'Think about user navigation and perspective'
    ]
  }
  return tips[mode] || []
}