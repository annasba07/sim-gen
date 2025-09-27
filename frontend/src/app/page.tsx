"use client"

import React, { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { SketchCanvas } from '@/components/sketch-canvas'
import { SimulationViewer } from '@/components/simulation-viewer'
import { Button } from '@/components/ui/button'
import { getPhysicsAPI } from '@/lib/physics-api'
import {
  Sparkles,
  Rocket,
  Wand2,
  Brain,
  Zap,
  ArrowRight,
  Palette,
  Cpu,
  Lightbulb,
  AlertCircle
} from 'lucide-react'

interface ProcessingStep {
  id: string
  title: string
  description: string
  progress: number
  isActive: boolean
  isComplete: boolean
}

export default function Home() {
  const [sketchData, setSketchData] = useState<string>("")
  const [textPrompt, setTextPrompt] = useState<string>("")
  const [isProcessing, setIsProcessing] = useState(false)
  const [mjcfResult, setMjcfResult] = useState<string>("")
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const physicsApi = getPhysicsAPI()
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    {
      id: 'analyze',
      title: 'Analyzing Sketch',
      description: 'AI vision understanding your drawing...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'enhance',
      title: 'Enhancing Prompt',
      description: 'Combining sketch with text description...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'generate',
      title: 'Generating Physics',
      description: 'Creating MuJoCo simulation...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'render',
      title: 'Rendering Simulation',
      description: 'Bringing your physics to life...',
      progress: 0,
      isActive: false,
      isComplete: false
    }
  ])

  const handleSketchChange = useCallback((dataURL: string) => {
    setSketchData(dataURL)
  }, [])

  const processWithRealAPI = async () => {
    setIsProcessing(true)
    setError(null)

    const steps = [...processingSteps]

    try {
      // Step 1: Analyzing Sketch
      steps[0].isActive = true
      setProcessingSteps([...steps])
      for (let progress = 0; progress <= 40; progress += 10) {
        steps[0].progress = progress
        setProcessingSteps([...steps])
        await new Promise(resolve => setTimeout(resolve, 50))
      }

      // Step 2: Enhancing Prompt
      steps[0].progress = 100
      steps[0].isActive = false
      steps[0].isComplete = true
      steps[1].isActive = true
      setProcessingSteps([...steps])

      for (let progress = 0; progress <= 40; progress += 10) {
        steps[1].progress = progress
        setProcessingSteps([...steps])
        await new Promise(resolve => setTimeout(resolve, 50))
      }

      // Step 3: Generate Physics - Real API Call
      steps[1].progress = 100
      steps[1].isActive = false
      steps[1].isComplete = true
      steps[2].isActive = true
      setProcessingSteps([...steps])

      // Make the actual API call
      const generateResponse = await physicsApi.generateFromPrompt({
        prompt: textPrompt || "Create an interesting physics simulation based on this sketch",
        sketch_data: sketchData ? sketchData.split(',')[1] : undefined, // Remove data:image/png;base64, prefix
        use_multimodal: true,
        max_bodies: 10,
        include_actuators: true,
        include_sensors: true
      })

      steps[2].progress = 100
      steps[2].isActive = false
      steps[2].isComplete = true

      if (generateResponse.success && generateResponse.mjcf_xml) {
        // Step 4: Rendering Simulation
        steps[3].isActive = true
        setProcessingSteps([...steps])

        for (let progress = 0; progress <= 100; progress += 20) {
          steps[3].progress = progress
          setProcessingSteps([...steps])
          await new Promise(resolve => setTimeout(resolve, 100))
        }

        steps[3].isActive = false
        steps[3].isComplete = true
        setProcessingSteps([...steps])

        // Set the real MJCF result
        setMjcfResult(generateResponse.mjcf_xml)
      } else {
        throw new Error(generateResponse.error || 'Failed to generate physics simulation')
      }
    } catch (err) {
      // Handle errors gracefully
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred'
      setError(errorMessage)
      console.error('API Error:', err)

      // Reset all steps on error
      steps.forEach(step => {
        step.isActive = false
        step.isComplete = false
        step.progress = 0
      })
      setProcessingSteps(steps)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleGenerate = async () => {
    if (!sketchData && !textPrompt) {
      setError('Please draw something or enter a text description!')
      return
    }

    await processWithRealAPI()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50">
      {/* Header */}
      <header className="relative overflow-hidden py-12 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="flex justify-center items-center gap-3 mb-4">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                SimGen AI
              </h1>
            </div>
            
            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-6">
              Transform your sketches into interactive physics simulations with the power of AI
            </p>
            
            <div className="flex justify-center gap-6 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4" />
                <span>Vision AI</span>
              </div>
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                <span>MuJoCo Physics</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                <span>Real-time 3D</span>
              </div>
            </div>
          </motion.div>
        </div>
      </header>

      {/* Main Interface */}
      <main className="max-w-7xl mx-auto px-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Input Section */}
          <motion.div 
            className="space-y-6"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Sketch Canvas */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Palette className="w-5 h-5 text-blue-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Draw Your Physics Idea</h2>
              </div>
              
              <SketchCanvas 
                width={600}
                height={400}
                onSketchChange={handleSketchChange}
                className="w-full"
              />
            </div>

            {/* Text Input */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Lightbulb className="w-5 h-5 text-purple-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Describe Your Vision</h2>
              </div>
              
              <textarea
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder="Describe what you want to happen... (e.g., 'make this robot arm pick up a red ball')"
                className="w-full h-24 p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700 placeholder-gray-400"
              />
            </div>

            {/* Error Display */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
                    <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-red-800 font-medium">Error</p>
                      <p className="text-red-700 text-sm mt-1">{error}</p>
                    </div>
                    <button
                      onClick={() => setError(null)}
                      className="ml-auto text-red-600 hover:text-red-700"
                    >
                      ×
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Generate Button */}
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Button
                onClick={handleGenerate}
                disabled={isProcessing}
                size="xl"
                variant="gradient"
                className="w-full text-lg font-semibold"
              >
                {isProcessing ? (
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Creating Physics Magic...</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <Wand2 className="w-5 h-5" />
                    <span>Generate Simulation</span>
                    <ArrowRight className="w-5 h-5" />
                  </div>
                )}
              </Button>
            </motion.div>
          </motion.div>

          {/* Output Section */}
          <motion.div 
            className="space-y-6"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            {/* Processing Steps */}
            <AnimatePresence>
              {isProcessing && (
                <motion.div 
                  className="space-y-4"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Rocket className="w-5 h-5 text-green-600" />
                    </div>
                    <h2 className="text-xl font-semibold text-gray-800">AI Processing</h2>
                  </div>
                  
                  <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 space-y-4">
                    {processingSteps.map((step, index) => (
                      <motion.div
                        key={step.id}
                        className={`flex items-center gap-4 p-3 rounded-lg transition-all duration-300 ${
                          step.isActive ? 'bg-blue-50 border-blue-200 border' : 
                          step.isComplete ? 'bg-green-50' : 'bg-gray-50'
                        }`}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                          step.isComplete ? 'bg-green-500 text-white' :
                          step.isActive ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-600'
                        }`}>
                          {step.isComplete ? '✓' : index + 1}
                        </div>
                        
                        <div className="flex-1">
                          <p className="font-medium text-gray-800">{step.title}</p>
                          <p className="text-sm text-gray-600">{step.description}</p>
                          
                          {step.isActive && (
                            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${step.progress}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Simulation Viewer */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Zap className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Interactive Physics</h2>
              </div>
              
              <SimulationViewer 
                mjcfContent={mjcfResult}
                className="w-full"
              />
            </div>
          </motion.div>
        </div>

        {/* Feature highlights */}
        <motion.div 
          className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          {[
            {
              icon: Brain,
              title: "AI Vision",
              description: "Advanced computer vision understands your sketches and converts them into physics descriptions"
            },
            {
              icon: Cpu,
              title: "Real Physics",
              description: "Powered by MuJoCo physics engine for realistic simulations with accurate dynamics"
            },
            {
              icon: Sparkles,
              title: "Interactive 3D",
              description: "Explore your simulations in real-time with full 3D controls and multiple viewing angles"
            }
          ].map((feature, index) => (
            <div key={index} className="text-center p-6 bg-white rounded-xl shadow-lg border border-gray-100">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl w-fit mx-auto mb-4">
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-gray-800 mb-2">{feature.title}</h3>
              <p className="text-gray-600 text-sm">{feature.description}</p>
            </div>
          ))}
        </motion.div>
      </main>
    </div>
  )
}
