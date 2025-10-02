"use client"

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Check, Sparkles, Lock } from 'lucide-react'

interface Mode {
  id: string
  name: string
  description: string
  icon: string
  color: string
  beta: boolean
  features: string[]
  available: boolean
}

interface ModeSelectorProps {
  onSelectMode: (modeId: string) => void
  selectedMode?: string
  className?: string
}

export function ModeSelector({ onSelectMode, selectedMode, className = "" }: ModeSelectorProps) {
  const [modes, setModes] = useState<Mode[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch available modes from API
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v2/modes`)
      .then(res => res.json())
      .then(data => {
        setModes(data.modes)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load modes:', err)
        // Fallback to static modes
        setModes(getFallbackModes())
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12">
        <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-3">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-full text-sm font-semibold"
        >
          <Sparkles className="w-4 h-4" />
          <span>VirtualForge</span>
        </motion.div>

        <h2 className="text-4xl font-bold text-gray-800">
          What do you want to create?
        </h2>
        <p className="text-gray-600 text-lg max-w-2xl mx-auto">
          Choose your creation mode and bring your ideas to life in seconds
        </p>
      </div>

      {/* Mode Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
        {modes.map((mode, index) => (
          <ModeCard
            key={mode.id}
            mode={mode}
            selected={selectedMode === mode.id}
            onSelect={() => mode.available && onSelectMode(mode.id)}
            index={index}
          />
        ))}
      </div>

      {/* Footer hint */}
      <p className="text-center text-sm text-gray-500">
        You can switch modes anytime • All your projects in one place
      </p>
    </div>
  )
}

interface ModeCardProps {
  mode: Mode
  selected: boolean
  onSelect: () => void
  index: number
}

function ModeCard({ mode, selected, onSelect, index }: ModeCardProps) {
  const colorMap: Record<string, string> = {
    blue: 'from-blue-500 to-cyan-500',
    purple: 'from-purple-500 to-pink-500',
    green: 'from-green-500 to-teal-500',
    orange: 'from-orange-500 to-red-500'
  }

  const gradient = colorMap[mode.color] || 'from-gray-500 to-gray-600'

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      whileHover={mode.available ? { scale: 1.05, y: -5 } : {}}
      onClick={onSelect}
      className={`
        relative p-6 rounded-2xl border-2 cursor-pointer transition-all
        ${selected
          ? 'border-purple-500 bg-purple-50 shadow-xl'
          : mode.available
          ? 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-lg'
          : 'border-gray-200 bg-gray-50 cursor-not-allowed opacity-60'
        }
      `}
    >
      {/* Selected indicator */}
      {selected && (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="absolute -top-2 -right-2 w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center shadow-lg"
        >
          <Check className="w-5 h-5 text-white" />
        </motion.div>
      )}

      {/* Beta/Coming Soon badge */}
      {mode.beta && mode.available && (
        <div className="absolute top-4 right-4 px-2 py-1 bg-yellow-100 text-yellow-700 text-xs font-semibold rounded-full">
          BETA
        </div>
      )}

      {!mode.available && (
        <div className="absolute top-4 right-4 px-2 py-1 bg-gray-200 text-gray-600 text-xs font-semibold rounded-full flex items-center gap-1">
          <Lock className="w-3 h-3" />
          SOON
        </div>
      )}

      {/* Icon */}
      <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${gradient} flex items-center justify-center text-4xl mb-4`}>
        {mode.icon}
      </div>

      {/* Content */}
      <div className="space-y-3">
        <h3 className="text-xl font-bold text-gray-800">{mode.name}</h3>
        <p className="text-gray-600 text-sm">{mode.description}</p>

        {/* Features */}
        <div className="space-y-1">
          {mode.features.slice(0, 3).map((feature, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-1.5 h-1.5 bg-gray-400 rounded-full" />
              <span>{formatFeature(feature)}</span>
            </div>
          ))}
        </div>

        {/* CTA */}
        {mode.available && (
          <button
            className={`
              w-full mt-4 py-2 rounded-lg font-semibold text-sm transition-all
              ${selected
                ? 'bg-purple-500 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }
            `}
          >
            {selected ? 'Selected' : 'Choose this mode'}
          </button>
        )}
      </div>
    </motion.div>
  )
}

function formatFeature(feature: string): string {
  return feature
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function getFallbackModes(): Mode[] {
  return [
    {
      id: 'physics',
      name: 'Physics Lab',
      description: 'Scientific simulations & education',
      icon: '🔬',
      color: 'blue',
      beta: false,
      features: ['sketch_analysis', '3d_visualization', 'educational_templates'],
      available: true
    },
    {
      id: 'games',
      name: 'Game Studio',
      description: '60-second game creation',
      icon: '🎮',
      color: 'purple',
      beta: true,
      features: ['instant_preview', 'remix_system', 'multi_engine_export'],
      available: true
    },
    {
      id: 'vr',
      name: 'VR Worlds',
      description: 'Immersive virtual experiences',
      icon: '🌐',
      color: 'green',
      beta: true,
      features: ['3d_modeling', 'vr_interactions', 'multiplayer_spaces'],
      available: false
    }
  ]
}