"use client";

import { useState } from 'react';
import { Sparkles, Gamepad2, Loader2, AlertCircle } from 'lucide-react';
import SketchCanvas from '../sketch-canvas';
import GamePreview from './game-preview';
import { gamesAPI, type GameSpec } from '@/lib/games-api';

interface GameCreatorProps {
  onBack?: () => void;
}

export default function GameCreator({ onBack }: GameCreatorProps) {
  const [prompt, setPrompt] = useState('');
  const [sketchData, setSketchData] = useState<string>('');
  const [gameType, setGameType] = useState<'platformer' | 'topdown' | 'puzzle' | 'shooter'>('platformer');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string>('');
  const [generatedGame, setGeneratedGame] = useState<{
    html: string;
    spec: GameSpec;
    title: string;
  } | null>(null);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please describe your game idea');
      return;
    }

    setIsGenerating(true);
    setError('');

    try {
      // Step 1: Generate game spec from prompt
      const generateResponse = await gamesAPI.generateGame({
        prompt,
        sketch_data: sketchData || undefined,
        gameType,
        complexity: 'simple',
      });

      if (!generateResponse.success || !generateResponse.game_spec) {
        setError(generateResponse.errors?.join(', ') || 'Failed to generate game');
        return;
      }

      // Step 2: Compile to Phaser HTML
      const compileResponse = await gamesAPI.compileGame({
        spec: generateResponse.game_spec,
        options: {
          minify: false,
          include_phaser: true,
          debug: false,
        },
      });

      if (!compileResponse.success || !compileResponse.html) {
        setError(compileResponse.errors?.join(', ') || 'Failed to compile game');
        return;
      }

      // Success!
      setGeneratedGame({
        html: compileResponse.html,
        spec: generateResponse.game_spec,
        title: generateResponse.game_spec.title || 'My Game',
      });
    } catch (err) {
      setError((err as Error).message || 'An unexpected error occurred');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSketchChange = (dataUrl: string) => {
    setSketchData(dataUrl);
  };

  const handleClearSketch = () => {
    setSketchData('');
  };

  // If game is generated, show preview
  if (generatedGame) {
    return (
      <div className="h-full">
        <GamePreview
          html={generatedGame.html}
          title={generatedGame.title}
          onClose={() => {
            setGeneratedGame(null);
            setPrompt('');
            setSketchData('');
          }}
        />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-purple-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <div className="p-6 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl">
              <Gamepad2 className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                Game Studio
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Create games in 60 seconds with AI
              </p>
            </div>
          </div>

          {onBack && (
            <button
              onClick={onBack}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              ← Back
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* Game Type Selection */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              1. Choose Game Type
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { value: 'platformer', label: 'Platformer', emoji: '🏃', desc: 'Jump & run' },
                { value: 'topdown', label: 'Top-down', emoji: '🎯', desc: 'Aerial view' },
                { value: 'puzzle', label: 'Puzzle', emoji: '🧩', desc: 'Brain teaser' },
                { value: 'shooter', label: 'Shooter', emoji: '🚀', desc: 'Pew pew' },
              ].map((type) => (
                <button
                  key={type.value}
                  onClick={() => setGameType(type.value as any)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    gameType === type.value
                      ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
                  }`}
                >
                  <div className="text-3xl mb-2">{type.emoji}</div>
                  <div className="font-semibold text-sm text-gray-900 dark:text-white">
                    {type.label}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">{type.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Sketch Canvas */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              2. Sketch Your Idea (Optional)
            </h3>
            <SketchCanvas
              onSketchChange={handleSketchChange}
              onClear={handleClearSketch}
              width={800}
              height={400}
            />
          </div>

          {/* Prompt Input */}
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">
              3. Describe Your Game
            </h3>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder={`Example: "A ${gameType} where you collect coins and avoid enemies. The player can double jump."`}
              className="w-full h-32 px-4 py-3 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
            />

            {/* Tips */}
            <div className="mt-4 p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <p className="text-sm text-purple-700 dark:text-purple-300 font-medium mb-2">
                💡 Tips for great games:
              </p>
              <ul className="text-sm text-purple-600 dark:text-purple-400 space-y-1 ml-4 list-disc">
                <li>Describe the player's abilities (jump, shoot, etc.)</li>
                <li>Mention what to collect or avoid</li>
                <li>Include win/lose conditions</li>
                <li>Keep it simple for best results</li>
              </ul>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-red-800 dark:text-red-300 mb-1">Generation Error</h4>
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            </div>
          )}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()}
            className="w-full py-4 px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 flex items-center justify-center gap-3 text-lg disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-6 h-6 animate-spin" />
                Generating Your Game...
              </>
            ) : (
              <>
                <Sparkles className="w-6 h-6" />
                Generate Game
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
