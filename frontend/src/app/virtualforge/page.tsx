"use client";

import { useState } from 'react';
import { Sparkles, ArrowLeft } from 'lucide-react';
import { ModeSelector } from '@/components/shared/mode-selector';
import GameCreator from '@/components/games/game-creator';
import { useMode } from '@/hooks/use-mode';

// Helper functions
function getModeIcon(mode: string) {
  const icons: Record<string, string> = {
    physics: '🔬',
    games: '🎮',
    vr: '🌐',
  };
  return icons[mode] || '✨';
}

function getModeName(mode: string) {
  const names: Record<string, string> = {
    physics: 'Physics Lab',
    games: 'Game Studio',
    vr: 'VR Worlds',
  };
  return names[mode] || 'Unknown';
}

export default function VirtualForgePage() {
  const { currentMode, setMode } = useMode();
  const [showModeSelector, setShowModeSelector] = useState(true);

  const handleModeSelect = (modeId: string) => {
    setMode(modeId as any);
    setShowModeSelector(false);
  };

  const handleBackToModeSelector = () => {
    setShowModeSelector(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-gray-950 dark:via-purple-950 dark:to-slate-950">
      {/* Header */}
      <header className="border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {!showModeSelector && (
              <button
                onClick={handleBackToModeSelector}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                aria-label="Back to mode selector"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </button>
            )}

            <div className="flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 dark:from-purple-400 dark:to-blue-400 bg-clip-text text-transparent">
                VirtualForge
              </h1>
            </div>

            {!showModeSelector && (
              <div className="flex items-center gap-2 ml-4 px-3 py-1 bg-purple-100 dark:bg-purple-900/30 rounded-full">
                <span className="text-2xl">{getModeIcon(currentMode)}</span>
                <span className="text-sm font-semibold text-purple-700 dark:text-purple-300">
                  {getModeName(currentMode)}
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://docs.virtualforge.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
            >
              Docs
            </a>
            <a
              href="https://github.com/virtualforge"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-gray-600 dark:text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="h-[calc(100vh-73px)]">
        {showModeSelector ? (
          /* Mode Selection Screen */
          <div className="h-full flex items-center justify-center p-6">
            <div className="max-w-6xl w-full">
              <div className="text-center mb-12">
                <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 dark:from-purple-400 dark:via-blue-400 dark:to-indigo-400 bg-clip-text text-transparent">
                  What do you want to create?
                </h2>
                <p className="text-lg text-gray-600 dark:text-gray-400">
                  Choose your creation mode and bring your ideas to life with AI
                </p>
              </div>

              <ModeSelector onSelect={handleModeSelect} />

              <div className="mt-12 text-center">
                <p className="text-sm text-gray-500 dark:text-gray-500">
                  ✨ Powered by Claude AI • Phaser 3 • MuJoCo • Three.js
                </p>
              </div>
            </div>
          </div>
        ) : (
          /* Creation Interface - Mode-specific */
          <div className="h-full">
            {currentMode === 'physics' && (
              <PhysicsCreator onBack={handleBackToModeSelector} />
            )}

            {currentMode === 'games' && (
              <GameCreator onBack={handleBackToModeSelector} />
            )}

            {currentMode === 'vr' && (
              <VRComingSoon onBack={handleBackToModeSelector} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

// Placeholder for Physics Creator (you already have this)
function PhysicsCreator({ onBack }: { onBack: () => void }) {
  return (
    <div className="h-full flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-blue-950">
      <div className="text-center p-8">
        <div className="text-6xl mb-4">🔬</div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Physics Lab
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          Physics creation UI coming soon!
        </p>
        <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
          (Use the main app page for physics simulations for now)
        </p>
        <button
          onClick={onBack}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-colors"
        >
          Back to Modes
        </button>
      </div>
    </div>
  );
}

// VR Coming Soon placeholder
function VRComingSoon({ onBack }: { onBack: () => void }) {
  return (
    <div className="h-full flex items-center justify-center bg-gradient-to-br from-green-50 to-teal-50 dark:from-gray-900 dark:to-green-950">
      <div className="text-center p-8">
        <div className="text-6xl mb-4">🌐</div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          VR Worlds
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Coming Soon in Q2 2025
        </p>
        <div className="inline-block px-4 py-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-sm font-medium mb-6">
          🚧 Under Development
        </div>
        <div className="max-w-md mx-auto mb-6">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Create immersive VR experiences with spatial audio, interactions, and multiplayer spaces.
          </p>
        </div>
        <button
          onClick={onBack}
          className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition-colors"
        >
          Back to Modes
        </button>
      </div>
    </div>
  );
}
