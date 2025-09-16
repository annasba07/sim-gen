'use client'

import React, { useState, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { SketchCanvas } from '@/components/sketch-canvas'
import { PhysicsViewer } from '@/components/physics-viewer'
import { ActuatorControls, ActuatorInfo } from '@/components/actuator-controls'
import { Button } from '@/components/ui/button'
import { getPhysicsAPI, PhysicsSpec } from '@/lib/physics-api'
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
  FileJson,
  Code,
  Download
} from 'lucide-react'

interface ProcessingStep {
  id: string
  title: string
  description: string
  progress: number
  isActive: boolean
  isComplete: boolean
}

export default function HomeV2() {
  const [sketchData, setSketchData] = useState<string>('')
  const [textPrompt, setTextPrompt] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [mjcfResult, setMjcfResult] = useState<string>('')
  const [physicsSpec, setPhysicsSpec] = useState<PhysicsSpec | null>(null)
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [actuators, setActuators] = useState<ActuatorInfo[]>([])
  const [showDebugInfo, setShowDebugInfo] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const physicsAPI = getPhysicsAPI()

  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([
    {
      id: 'analyze',
      title: 'Analyzing Input',
      description: 'Processing sketch and text...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'generate',
      title: 'Generating PhysicsSpec',
      description: 'Creating structured physics description...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'compile',
      title: 'Compiling to MJCF',
      description: 'Converting to MuJoCo XML format...',
      progress: 0,
      isActive: false,
      isComplete: false
    },
    {
      id: 'validate',
      title: 'Validating Physics',
      description: 'Ensuring physical validity...',
      progress: 0,
      isActive: false,
      isComplete: false
    }
  ])

  const handleSketchChange = useCallback((dataURL: string) => {
    setSketchData(dataURL)
  }, [])

  const updateProcessingStep = (stepId: string, updates: Partial<ProcessingStep>) => {
    setProcessingSteps(prev =>
      prev.map(step =>
        step.id === stepId ? { ...step, ...updates } : step
      )
    )
  }

  const handleGenerate = async () => {
    if (!sketchData && !textPrompt && !selectedTemplate) {
      alert('Please draw something, enter a description, or select a template!')
      return
    }

    setIsProcessing(true)
    const steps = [...processingSteps]

    try {
      // Step 1: Analyze input
      updateProcessingStep('analyze', { isActive: true, progress: 0 })
      await simulateProgress('analyze', 500)

      let spec: PhysicsSpec | null = null
      let mjcf: string = ''

      if (selectedTemplate) {
        // Load template
        const result = await physicsAPI.loadTemplate(selectedTemplate)
        if (result.success && result.mjcf_xml && result.spec) {
          spec = result.spec
          mjcf = result.mjcf_xml
        }
      } else {
        // Step 2: Generate PhysicsSpec
        updateProcessingStep('generate', { isActive: true, progress: 0 })

        const generateResult = await physicsAPI.generateFromPrompt({
          prompt: textPrompt || 'Create a simple physics simulation',
          sketch_data: sketchData ? sketchData.split(',')[1] : undefined,
          use_multimodal: !!sketchData,
          include_actuators: true,
          include_sensors: true
        })

        await simulateProgress('generate', 1000)

        if (generateResult.success && generateResult.physics_spec) {
          spec = generateResult.physics_spec
          setPhysicsSpec(spec)
        }

        // Step 3: Compile to MJCF
        if (spec) {
          updateProcessingStep('compile', { isActive: true, progress: 0 })

          const compileResult = await physicsAPI.compileSpec({
            spec,
            validate_spec: true
          })

          await simulateProgress('compile', 500)

          if (compileResult.success && compileResult.mjcf_xml) {
            mjcf = compileResult.mjcf_xml
          }
        }
      }

      // Step 4: Validate
      updateProcessingStep('validate', { isActive: true, progress: 0 })
      await simulateProgress('validate', 300)

      // Set results
      if (mjcf) {
        setMjcfResult(mjcf)
        setPhysicsSpec(spec)

        // Extract actuators from spec
        if (spec?.actuators) {
          const actuatorInfos: ActuatorInfo[] = spec.actuators.map((a: any) => ({
            id: a.id,
            name: a.id,
            type: a.type || 'motor',
            ctrlrange: a.ctrlrange,
            forcerange: a.forcerange,
            currentValue: 0
          }))
          setActuators(actuatorInfos)
        }
      }

      // Mark all complete
      setProcessingSteps(prev =>
        prev.map(step => ({ ...step, isActive: false, isComplete: true, progress: 100 }))
      )

    } catch (error) {
      console.error('Generation failed:', error)
      alert('Failed to generate simulation. Please try again.')
    } finally {
      setIsProcessing(false)
    }
  }

  const simulateProgress = async (stepId: string, duration: number) => {
    const steps = 10
    const delay = duration / steps

    for (let i = 0; i <= steps; i++) {
      updateProcessingStep(stepId, { progress: (i / steps) * 100 })
      await new Promise(resolve => setTimeout(resolve, delay))
    }

    updateProcessingStep(stepId, { isActive: false, isComplete: true })
  }

  const handleTemplateSelect = async (templateName: string) => {
    setSelectedTemplate(templateName)
    setTextPrompt(`Load ${templateName} template`)
  }

  const downloadSpec = () => {
    if (!physicsSpec) return

    const blob = new Blob([JSON.stringify(physicsSpec, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${physicsSpec.meta.name || 'physics'}_spec.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadMJCF = () => {
    if (!mjcfResult) return

    const blob = new Blob([mjcfResult], { type: 'text/xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${physicsSpec?.meta.name || 'physics'}.xml`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleControlChange = (values: Float32Array) => {
    // This will be handled by the PhysicsViewer WebSocket connection
    console.log('Control values:', values)
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
                SimGen AI v2
              </h1>
            </div>

            <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-6">
              PhysicsSpec Architecture - Reliable physics generation with binary streaming
            </p>

            <div className="flex justify-center gap-6 text-sm text-gray-500">
              <div className="flex items-center gap-2">
                <FileJson className="w-4 h-4" />
                <span>PhysicsSpec</span>
              </div>
              <div className="flex items-center gap-2">
                <Code className="w-4 h-4" />
                <span>MJCF Compiler</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4" />
                <span>Binary Stream</span>
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
            {/* Template Selection */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Cpu className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Quick Templates</h2>
              </div>

              <div className="grid grid-cols-2 gap-2">
                {['pendulum', 'double_pendulum', 'cart_pole', 'robot_arm'].map(template => (
                  <button
                    key={template}
                    onClick={() => handleTemplateSelect(template)}
                    className={`p-3 rounded-lg border transition-all ${
                      selectedTemplate === template
                        ? 'border-blue-500 bg-blue-50 text-blue-700'
                        : 'border-gray-300 hover:border-gray-400 text-gray-700'
                    }`}
                  >
                    {template.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </button>
                ))}
              </div>
            </div>

            {/* Sketch Canvas */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Palette className="w-5 h-5 text-blue-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Draw Your Physics</h2>
              </div>

              <SketchCanvas
                ref={canvasRef as any}
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
                placeholder="Describe your physics scenario... (e.g., 'robot arm that can pick up objects')"
                className="w-full h-24 p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700 placeholder-gray-400"
              />
            </div>

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
                    <span>Generating Physics...</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-3">
                    <Wand2 className="w-5 h-5" />
                    <span>Generate with PhysicsSpec</span>
                    <ArrowRight className="w-5 h-5" />
                  </div>
                )}
              </Button>
            </motion.div>

            {/* Debug Controls */}
            {physicsSpec && (
              <div className="flex gap-2">
                <button
                  onClick={() => setShowDebugInfo(!showDebugInfo)}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition-colors"
                >
                  {showDebugInfo ? 'Hide' : 'Show'} Debug Info
                </button>
                <button
                  onClick={downloadSpec}
                  className="px-4 py-2 bg-blue-100 hover:bg-blue-200 rounded-lg text-sm transition-colors flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  PhysicsSpec JSON
                </button>
                <button
                  onClick={downloadMJCF}
                  className="px-4 py-2 bg-green-100 hover:bg-green-200 rounded-lg text-sm transition-colors flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  MJCF XML
                </button>
              </div>
            )}
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
                    <h2 className="text-xl font-semibold text-gray-800">PhysicsSpec Pipeline</h2>
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

            {/* Physics Viewer */}
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Zap className="w-5 h-5 text-indigo-600" />
                </div>
                <h2 className="text-xl font-semibold text-gray-800">Real-time Physics (Binary Stream)</h2>
              </div>

              <PhysicsViewer
                mjcfContent={mjcfResult}
                physicsSpec={physicsSpec}
                autoStart={true}
                className="w-full"
              />
            </div>

            {/* Actuator Controls */}
            {actuators.length > 0 && (
              <ActuatorControls
                actuators={actuators}
                onControlChange={handleControlChange}
                className="w-full"
              />
            )}

            {/* Debug Info */}
            {showDebugInfo && physicsSpec && (
              <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-xs overflow-auto max-h-96">
                <div className="mb-2 text-yellow-400">// PhysicsSpec JSON</div>
                <pre>{JSON.stringify(physicsSpec, null, 2)}</pre>
                {mjcfResult && (
                  <>
                    <div className="mt-4 mb-2 text-yellow-400">// Compiled MJCF XML</div>
                    <pre className="text-gray-300">{mjcfResult.substring(0, 500)}...</pre>
                  </>
                )}
              </div>
            )}
          </motion.div>
        </div>
      </main>
    </div>
  )
}