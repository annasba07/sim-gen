"use client";

import { useState, useRef, useEffect } from 'react';
import { Play, Pause, RotateCcw, Maximize2, Download, Share2, Code } from 'lucide-react';

interface GamePreviewProps {
  html: string;
  title?: string;
  onClose?: () => void;
}

export default function GamePreview({ html, title = "Game Preview", onClose }: GamePreviewProps) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Handle fullscreen
  const toggleFullscreen = async () => {
    if (!containerRef.current) return;

    if (!document.fullscreenElement) {
      await containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      await document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  // Play/Pause (reload iframe to pause)
  const togglePlay = () => {
    if (isPlaying) {
      // Pause by blanking the iframe
      if (iframeRef.current) {
        iframeRef.current.src = 'about:blank';
      }
    } else {
      // Resume by reloading
      loadGame();
    }
    setIsPlaying(!isPlaying);
  };

  // Restart game
  const restart = () => {
    loadGame();
    setIsPlaying(true);
  };

  // Load game HTML into iframe
  const loadGame = () => {
    if (iframeRef.current) {
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      iframeRef.current.src = url;
    }
  };

  // Download game as HTML file
  const downloadGame = () => {
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title.toLowerCase().replace(/\s+/g, '-')}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Share game (copy link - would need CDN in production)
  const shareGame = async () => {
    try {
      await navigator.clipboard.writeText(html);
      alert('Game HTML copied to clipboard! You can paste it to share.');
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  // Initial load
  useEffect(() => {
    loadGame();
  }, [html]);

  return (
    <div
      ref={containerRef}
      className={`flex flex-col ${isFullscreen ? 'h-screen w-screen' : 'h-full'} bg-gray-900`}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>

        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>

          {/* Restart */}
          <button
            onClick={restart}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Restart"
          >
            <RotateCcw className="w-5 h-5" />
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Fullscreen"
          >
            <Maximize2 className="w-5 h-5" />
          </button>

          {/* Divider */}
          <div className="w-px h-6 bg-gray-600 mx-2"></div>

          {/* Download */}
          <button
            onClick={downloadGame}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Download HTML"
          >
            <Download className="w-5 h-5" />
          </button>

          {/* Share */}
          <button
            onClick={shareGame}
            className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors"
            title="Share"
          >
            <Share2 className="w-5 h-5" />
          </button>

          {/* View Code */}
          <button
            onClick={() => setShowCode(!showCode)}
            className={`p-2 rounded-lg ${
              showCode ? 'bg-purple-600' : 'bg-gray-700'
            } hover:bg-purple-500 text-white transition-colors`}
            title="View Code"
          >
            <Code className="w-5 h-5" />
          </button>

          {/* Close */}
          {onClose && (
            <>
              <div className="w-px h-6 bg-gray-600 mx-2"></div>
              <button
                onClick={onClose}
                className="px-3 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-colors text-sm font-medium"
              >
                Close
              </button>
            </>
          )}
        </div>
      </div>

      {/* Game Preview or Code View */}
      <div className="flex-1 relative overflow-hidden">
        {showCode ? (
          <div className="absolute inset-0 overflow-auto p-4 bg-gray-900">
            <pre className="text-sm text-gray-300 font-mono">
              <code>{html}</code>
            </pre>
          </div>
        ) : (
          <iframe
            ref={iframeRef}
            className="w-full h-full border-0"
            title={title}
            sandbox="allow-scripts allow-same-origin"
          />
        )}
      </div>

      {/* Footer Stats */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-t border-gray-700 text-sm text-gray-400">
        <div className="flex items-center gap-4">
          <span>Size: {(html.length / 1024).toFixed(2)} KB</span>
          <span>•</span>
          <span>Powered by Phaser 3</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs">
            Made with <span className="text-purple-400">VirtualForge</span>
          </span>
        </div>
      </div>
    </div>
  );
}
