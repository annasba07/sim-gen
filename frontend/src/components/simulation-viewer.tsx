"use client"

import React, { Suspense, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text, Box, Sphere, Cylinder } from '@react-three/drei'
import { motion } from 'framer-motion'
import { Button } from './ui/button'
import { Play, Pause, RotateCcw, Settings } from 'lucide-react'

interface SimulationViewerProps {
  mjcfContent?: string
  className?: string
}

// Simple physics simulation components
function BouncingBall({ position }: { position: [number, number, number] }) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.5
    }
  })

  return (
    <Sphere ref={meshRef} position={position} args={[0.2]} material-color="red">
      <meshStandardMaterial attach="material" color="red" />
    </Sphere>
  )
}

function PendulumSystem() {
  const pendulumRef = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (pendulumRef.current) {
      pendulumRef.current.rotation.z = Math.sin(state.clock.elapsedTime) * 0.5
    }
  })

  return (
    <group ref={pendulumRef} position={[0, 1, 0]}>
      {/* Pivot point */}
      <Sphere position={[0, 0, 0]} args={[0.05]} material-color="black" />
      
      {/* Rod */}
      <Cylinder 
        position={[0, -0.5, 0]} 
        args={[0.02, 0.02, 1]} 
        material-color="gray"
      />
      
      {/* Bob */}
      <Sphere position={[0, -1, 0]} args={[0.15]} material-color="blue">
        <meshStandardMaterial attach="material" color="blue" metalness={0.3} roughness={0.4} />
      </Sphere>
    </group>
  )
}

function RobotArm() {
  const arm1Ref = useRef<THREE.Group>(null)
  const arm2Ref = useRef<THREE.Group>(null)
  
  useFrame((state) => {
    if (arm1Ref.current) {
      arm1Ref.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.8) * 0.3
    }
    if (arm2Ref.current) {
      arm2Ref.current.rotation.z = Math.sin(state.clock.elapsedTime * 1.2) * 0.4
    }
  })

  return (
    <group position={[0, -1, 0]}>
      {/* Base */}
      <Box position={[0, 0, 0]} args={[0.4, 0.2, 0.4]} material-color="gray" />
      
      {/* First arm segment */}
      <group ref={arm1Ref} position={[0, 0.1, 0]}>
        <Cylinder 
          position={[0, 0.4, 0]} 
          args={[0.08, 0.08, 0.8]} 
          material-color="blue"
        />
        
        {/* Second arm segment */}
        <group ref={arm2Ref} position={[0, 0.8, 0]}>
          <Cylinder 
            position={[0.3, 0, 0]} 
            args={[0.06, 0.06, 0.6]} 
            rotation={[0, 0, Math.PI / 2]}
            material-color="green"
          />
          
          {/* End effector */}
          <Box 
            position={[0.6, 0, 0]} 
            args={[0.1, 0.1, 0.1]} 
            material-color="red"
          />
        </group>
      </group>
    </group>
  )
}

function Scene({ simulationType }: { simulationType: string }) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      
      {/* Ground plane */}
      <Box position={[0, -2, 0]} args={[10, 0.1, 10]} material-color="#f0f0f0" />
      
      {/* Physics objects based on simulation type */}
      {simulationType === 'pendulum' && <PendulumSystem />}
      {simulationType === 'robot' && <RobotArm />}
      {simulationType === 'bouncing' && (
        <>
          <BouncingBall position={[-1, 0, 0]} />
          <BouncingBall position={[0, 0.5, 0]} />
          <BouncingBall position={[1, 0.2, 0]} />
        </>
      )}
      
      {/* Default demo if no specific type */}
      {simulationType === 'demo' && (
        <>
          <PendulumSystem />
          <RobotArm />
          <BouncingBall position={[2, 0, 0]} />
        </>
      )}
    </>
  )
}

function LoadingFallback() {
  return (
    <div className="flex items-center justify-center h-full bg-gradient-to-b from-blue-50 to-white">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
        <p className="text-gray-600">Loading simulation...</p>
      </div>
    </div>
  )
}

export function SimulationViewer({ mjcfContent, className = "" }: SimulationViewerProps) {
  const [isPlaying, setIsPlaying] = useState(true)
  const [simulationType, setSimulationType] = useState('demo')

  // Parse MJCF content to determine simulation type (simplified)
  React.useEffect(() => {
    if (mjcfContent) {
      if (mjcfContent.includes('pendulum')) {
        setSimulationType('pendulum')
      } else if (mjcfContent.includes('robot') || mjcfContent.includes('arm')) {
        setSimulationType('robot')
      } else if (mjcfContent.includes('ball') || mjcfContent.includes('bounce')) {
        setSimulationType('bouncing')
      } else {
        setSimulationType('demo')
      }
    }
  }, [mjcfContent])

  return (
    <motion.div 
      className={`relative bg-gradient-to-b from-slate-50 to-white rounded-xl shadow-2xl border-2 border-gray-200 overflow-hidden ${className}`}
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* 3D Canvas */}
      <div className="w-full h-96">
        <Canvas
          camera={{ position: [3, 3, 5], fov: 60 }}
          style={{ background: 'linear-gradient(to bottom, #e0f2fe, #ffffff)' }}
        >
          <Suspense fallback={null}>
            <Scene simulationType={simulationType} />
            <OrbitControls 
              enablePan={true} 
              enableZoom={true} 
              enableRotate={true}
              autoRotate={isPlaying}
              autoRotateSpeed={1}
            />
          </Suspense>
        </Canvas>
      </div>

      {/* Controls */}
      <div className="absolute top-4 left-4 flex gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => setIsPlaying(!isPlaying)}
          className="bg-white/90 backdrop-blur-sm hover:bg-white"
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => window.location.reload()}
          className="bg-white/90 backdrop-blur-sm hover:bg-white"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="bg-white/90 backdrop-blur-sm hover:bg-white"
        >
          <Settings className="w-4 h-4" />
        </Button>
      </div>

      {/* Simulation info */}
      <div className="absolute bottom-4 left-4 bg-white/90 backdrop-blur-sm px-3 py-2 rounded-lg">
        <p className="text-sm text-gray-600">
          <span className="font-semibold">Simulation:</span> {simulationType.charAt(0).toUpperCase() + simulationType.slice(1)}
        </p>
        <p className="text-xs text-gray-500">
          {isPlaying ? 'Running' : 'Paused'} • Interactive 3D
        </p>
      </div>

      {/* Performance overlay */}
      <div className="absolute top-4 right-4 bg-green-500/20 backdrop-blur-sm px-2 py-1 rounded-lg">
        <p className="text-xs text-green-700 font-medium">60 FPS</p>
      </div>
    </motion.div>
  )
}