"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { SketchCanvas } from './sketch-canvas'
import { Button } from './ui/button'
import {
  Gamepad2,
  Sparkles,
  Wand2,
  Trophy,
  Users,
  Zap,
  BookOpen,
  Car,
  Puzzle,
  Sword,
  Heart,
  Target,
  Building,
  Cpu,
  Globe,
  Code,
  PlayCircle
} from 'lucide-react'

interface GameTemplate {
  id: string
  name: string
  icon: React.ReactNode
  description: string
  prompt: string
  color: string
}

interface GameGeneratorProps {
  className?: string
}

const gameTemplates: GameTemplate[] = [
  {
    id: 'platformer',
    name: 'Platformer',
    icon: <Gamepad2 className="w-5 h-5" />,
    description: 'Jump and run adventure',
    prompt: 'Create a platformer where the player jumps across platforms collecting coins',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    id: 'puzzle',
    name: 'Puzzle',
    icon: <Puzzle className="w-5 h-5" />,
    description: 'Brain-teasing challenges',
    prompt: 'Make a physics puzzle game with movable blocks and goals',
    color: 'from-purple-500 to-pink-500'
  },
  {
    id: 'survival',
    name: 'Survival',
    icon: <Heart className="w-5 h-5" />,
    description: 'Stay alive and thrive',
    prompt: 'Build a survival game where you gather resources and defend against enemies',
    color: 'from-red-500 to-orange-500'
  },
  {
    id: 'educational',
    name: 'Educational',
    icon: <BookOpen className="w-5 h-5" />,
    description: 'Learn through play',
    prompt: 'Create an educational game that teaches basic math concepts',
    color: 'from-green-500 to-teal-500'
  },
  {
    id: 'racing',
    name: 'Racing',
    icon: <Car className="w-5 h-5" />,
    description: 'Speed and competition',
    prompt: 'Design a racing game with obstacles and power-ups',
    color: 'from-yellow-500 to-amber-500'
  },
  {
    id: 'rpg',
    name: 'RPG',
    icon: <Sword className="w-5 h-5" />,
    description: 'Adventure and quests',
    prompt: 'Make an RPG with a hero who explores dungeons and fights monsters',
    color: 'from-indigo-500 to-purple-500'
  }
]

const engineOptions = [
  {
    id: 'babylon',
    name: 'Web (Babylon.js)',
    icon: <Globe className="w-4 h-4" />,
    description: 'Instant browser play'
  },
  {
    id: 'unity',
    name: 'Unity',
    icon: <Cpu className="w-4 h-4" />,
    description: 'Professional quality'
  },
  {
    id: 'roblox',
    name: 'Roblox',
    icon: <Users className="w-4 h-4" />,
    description: 'Social gaming'
  },
  {
    id: 'mujoco',
    name: 'Physics Sim',
    icon: <Zap className="w-4 h-4" />,
    description: 'Realistic physics'
  }
]

export function GameGenerator({ className = "" }: GameGeneratorProps) {
  const [prompt, setPrompt] = useState('')
  const [selectedTemplate, setSelectedTemplate] = useState<GameTemplate | null>(null)
  const [selectedEngine, setSelectedEngine] = useState('babylon')
  const [sketchData, setSketchData] = useState<string>('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [generatedGame, setGeneratedGame] = useState<any>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleTemplateSelect = (template: GameTemplate) => {
    setSelectedTemplate(template)
    setPrompt(template.prompt)
  }

  const handleGenerate = async () => {
    setIsGenerating(true)

    // Simulate game generation (would call real API)
    setTimeout(() => {
      setGeneratedGame({
        name: 'Your Generated Game',
        playUrl: 'https://play.simgen.ai/preview/game-123',
        engineOutput: selectedEngine,
        specs: {
          entities: 12,
          mechanics: 5,
          levels: 3
        }
      })
      setIsGenerating(false)
    }, 3000)
  }

  return (
    <div className={`space-y-8 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-3 px-4 py-2 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-full"
        >
          <Sparkles className="w-5 h-5" />
          <span className="font-semibold">AI Game Creator</span>
          <span className="text-xs bg-white/20 px-2 py-1 rounded-full">BETA</span>
        </motion.div>

        <h2 className="text-3xl font-bold text-gray-800">
          From Idea to Playable Game in Seconds
        </h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Describe your game idea or sketch it out, and our AI will generate a fully playable game
          with mechanics, levels, and objectives.
        </p>
      </div>

      {/* Game Templates */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Choose a Template or Start Fresh</h3>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {gameTemplates.map((template) => (
            <motion.button
              key={template.id}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleTemplateSelect(template)}
              className={`relative p-4 rounded-xl border-2 transition-all ${
                selectedTemplate?.id === template.id
                  ? 'border-purple-500 bg-purple-50'
                  : 'border-gray-200 hover:border-gray-300 bg-white'
              }`}
            >
              <div
                className={`w-10 h-10 rounded-lg bg-gradient-to-r ${template.color}
                  flex items-center justify-center text-white mx-auto mb-2`}
              >
                {template.icon}
              </div>
              <p className="text-sm font-medium text-gray-800">{template.name}</p>
              <p className="text-xs text-gray-500 mt-1">{template.description}</p>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Main Input Area */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Text Prompt */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Wand2 className="w-5 h-5 text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-700">Describe Your Game</h3>
          </div>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Example: Create a puzzle platformer where a robot must navigate through a factory using magnets to move metal platforms and reach the exit..."
            className="w-full h-48 p-4 border-2 border-gray-200 rounded-xl focus:ring-2
              focus:ring-purple-500 focus:border-transparent resize-none text-gray-700"
          />

          {/* Engine Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-600">Target Platform</label>
            <div className="grid grid-cols-2 gap-2">
              {engineOptions.map((engine) => (
                <button
                  key={engine.id}
                  onClick={() => setSelectedEngine(engine.id)}
                  className={`p-3 rounded-lg border-2 transition-all flex items-center gap-2 ${
                    selectedEngine === engine.id
                      ? 'border-purple-500 bg-purple-50'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                  }`}
                >
                  {engine.icon}
                  <div className="text-left">
                    <p className="text-sm font-medium">{engine.name}</p>
                    <p className="text-xs text-gray-500">{engine.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Sketch Canvas */}
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-purple-600" />
            <h3 className="text-lg font-semibold text-gray-700">Sketch Your Level (Optional)</h3>
          </div>

          <SketchCanvas
            width={500}
            height={300}
            onSketchChange={setSketchData}
            className="w-full"
          />
        </div>
      </div>

      {/* Advanced Options */}
      <div className="border-t border-gray-200 pt-6">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-purple-600 font-medium text-sm hover:text-purple-700"
        >
          {showAdvanced ? '− Hide' : '+ Show'} Advanced Options
        </button>

        <AnimatePresence>
          {showAdvanced && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4"
            >
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">Include Tutorial</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">Multiplayer Support</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">Add Boss Battles</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded" />
                <span className="text-sm">Procedural Levels</span>
              </label>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Generate Button */}
      <div className="flex justify-center">
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="w-full max-w-md"
        >
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !prompt}
            size="lg"
            className="w-full bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600
              hover:to-purple-700 text-white font-semibold py-4 text-lg"
          >
            {isGenerating ? (
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Creating Your Game...</span>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5" />
                <span>Generate Game</span>
                <PlayCircle className="w-5 h-5" />
              </div>
            )}
          </Button>
        </motion.div>
      </div>

      {/* Generated Game Preview */}
      <AnimatePresence>
        {generatedGame && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mt-8 p-6 bg-gradient-to-r from-violet-50 to-purple-50 rounded-2xl border-2 border-purple-200"
          >
            <div className="flex items-start justify-between">
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-white rounded-xl shadow-md">
                    <Trophy className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800">{generatedGame.name}</h3>
                    <p className="text-sm text-gray-600">Ready to play!</p>
                  </div>
                </div>

                <div className="flex gap-6 text-sm">
                  <div>
                    <span className="text-gray-500">Entities:</span>
                    <span className="ml-2 font-semibold">{generatedGame.specs.entities}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Mechanics:</span>
                    <span className="ml-2 font-semibold">{generatedGame.specs.mechanics}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Levels:</span>
                    <span className="ml-2 font-semibold">{generatedGame.specs.levels}</span>
                  </div>
                </div>

                <div className="flex gap-3">
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-green-500 to-emerald-600 text-white"
                  >
                    <PlayCircle className="w-5 h-5 mr-2" />
                    Play Now
                  </Button>
                  <Button
                    size="lg"
                    variant="outline"
                  >
                    <Code className="w-5 h-5 mr-2" />
                    View Code
                  </Button>
                  <Button
                    size="lg"
                    variant="outline"
                  >
                    <Building className="w-5 h-5 mr-2" />
                    Edit Game
                  </Button>
                </div>
              </div>

              <div className="hidden lg:block">
                <div className="w-64 h-48 bg-white rounded-xl shadow-lg flex items-center justify-center">
                  <p className="text-gray-400">Game Preview</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}