/**
 * Physics Renderer: Three.js integration for MuJoCo physics visualization
 * Renders physics bodies based on streaming frame data
 */

import * as THREE from 'three'
import { OrbitControls } from '@react-three/drei'
import { PhysicsFrame, ModelManifest } from './physics-client'

export interface BodyMesh {
  bodyId: number
  name: string
  mesh: THREE.Mesh | THREE.Group
  geomType: string
  initialPosition: THREE.Vector3
  initialQuaternion: THREE.Quaternion
}

export interface InterpolationState {
  enabled: boolean
  alpha: number
  previousFrame: PhysicsFrame | null
  currentFrame: PhysicsFrame | null
  nextFrame: PhysicsFrame | null
  lastUpdateTime: number
}

export class PhysicsRenderer {
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: any // OrbitControls type
  private bodies: Map<number, BodyMesh> = new Map()
  private manifest: ModelManifest | null = null
  private interpolation: InterpolationState = {
    enabled: true,
    alpha: 0,
    previousFrame: null,
    currentFrame: null,
    nextFrame: null,
    lastUpdateTime: 0,
  }
  private frameRate: number = 60
  private animationId: number | null = null

  constructor(
    container: HTMLElement,
    width?: number,
    height?: number
  ) {
    // Create scene
    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0xf0f0f0)
    this.scene.fog = new THREE.Fog(0xf0f0f0, 10, 50)

    // Create camera
    const w = width || container.clientWidth
    const h = height || container.clientHeight
    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 100)
    this.camera.position.set(2, 2, 4)
    this.camera.lookAt(0, 0, 0)

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    })
    this.renderer.setSize(w, h)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap
    container.appendChild(this.renderer.domElement)

    // Add orbit controls
    const OrbitControlsImpl = require('three/examples/jsm/controls/OrbitControls').OrbitControls
    this.controls = new OrbitControlsImpl(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.minDistance = 0.5
    this.controls.maxDistance = 20

    // Add lights
    this.setupLighting()

    // Add ground plane
    this.addGroundPlane()

    // Add coordinate axes helper
    const axesHelper = new THREE.AxesHelper(1)
    this.scene.add(axesHelper)

    // Handle window resize
    window.addEventListener('resize', this.handleResize.bind(this))
  }

  /**
   * Setup scene lighting
   */
  private setupLighting(): void {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    this.scene.add(ambientLight)

    // Directional light (sun)
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4)
    directionalLight.position.set(5, 10, 5)
    directionalLight.castShadow = true
    directionalLight.shadow.camera.left = -10
    directionalLight.shadow.camera.right = 10
    directionalLight.shadow.camera.top = 10
    directionalLight.shadow.camera.bottom = -10
    directionalLight.shadow.camera.near = 0.1
    directionalLight.shadow.camera.far = 50
    directionalLight.shadow.mapSize.width = 2048
    directionalLight.shadow.mapSize.height = 2048
    this.scene.add(directionalLight)

    // Hemisphere light for better ambient
    const hemisphereLight = new THREE.HemisphereLight(0x87ceeb, 0x98d8c8, 0.3)
    this.scene.add(hemisphereLight)
  }

  /**
   * Add ground plane
   */
  private addGroundPlane(): void {
    const geometry = new THREE.PlaneGeometry(20, 20)
    const material = new THREE.MeshStandardMaterial({
      color: 0xe0e0e0,
      roughness: 0.8,
      metalness: 0.2,
    })
    const ground = new THREE.Mesh(geometry, material)
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    this.scene.add(ground)

    // Add grid
    const gridHelper = new THREE.GridHelper(20, 20, 0x888888, 0xcccccc)
    gridHelper.position.y = 0.001
    this.scene.add(gridHelper)
  }

  /**
   * Initialize bodies from manifest
   */
  initializeFromManifest(manifest: ModelManifest): void {
    this.manifest = manifest

    // Clear existing bodies
    this.bodies.forEach(body => {
      this.scene.remove(body.mesh)
    })
    this.bodies.clear()

    // Create body meshes based on manifest
    for (let i = 0; i < manifest.nbody; i++) {
      const bodyMesh = this.createBodyMesh(
        i,
        manifest.body_names[i],
        manifest.geom_types[i] || 'box',
        manifest.geom_sizes[i] || [0.1, 0.1, 0.1],
        manifest.geom_rgba[i] || [0.7, 0.7, 0.7, 1.0]
      )

      if (bodyMesh) {
        this.bodies.set(i, bodyMesh)
        this.scene.add(bodyMesh.mesh)
      }
    }

    console.log(`[PhysicsRenderer] Initialized ${this.bodies.size} bodies`)
  }

  /**
   * Create mesh for body
   */
  private createBodyMesh(
    bodyId: number,
    name: string,
    geomType: string,
    size: number[],
    rgba: number[]
  ): BodyMesh | null {
    let geometry: THREE.BufferGeometry
    let mesh: THREE.Mesh

    // Skip ground plane (usually body 0)
    if (name === 'world' || name === 'ground') {
      return null
    }

    // Create geometry based on type
    switch (geomType) {
      case 'box':
      case 0: // GeomType enum value
        geometry = new THREE.BoxGeometry(
          size[0] * 2,
          size[1] * 2,
          size[2] * 2
        )
        break

      case 'sphere':
      case 2:
        geometry = new THREE.SphereGeometry(size[0], 32, 16)
        break

      case 'capsule':
      case 3:
        // Capsule approximated with cylinder + spheres
        const capsuleGroup = new THREE.Group()

        // Cylinder body
        const cylGeometry = new THREE.CylinderGeometry(
          size[0],
          size[0],
          size[1] || 0.5,
          16
        )
        const cylMaterial = new THREE.MeshStandardMaterial({
          color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
          opacity: rgba[3],
          transparent: rgba[3] < 1,
        })
        const cylinder = new THREE.Mesh(cylGeometry, cylMaterial)
        cylinder.castShadow = true
        cylinder.receiveShadow = true
        capsuleGroup.add(cylinder)

        // Top sphere
        const sphereGeometry = new THREE.SphereGeometry(size[0], 16, 8)
        const topSphere = new THREE.Mesh(sphereGeometry, cylMaterial)
        topSphere.position.y = (size[1] || 0.5) / 2
        capsuleGroup.add(topSphere)

        // Bottom sphere
        const bottomSphere = new THREE.Mesh(sphereGeometry, cylMaterial)
        bottomSphere.position.y = -(size[1] || 0.5) / 2
        capsuleGroup.add(bottomSphere)

        return {
          bodyId,
          name,
          mesh: capsuleGroup,
          geomType: 'capsule',
          initialPosition: new THREE.Vector3(),
          initialQuaternion: new THREE.Quaternion(),
        }

      case 'cylinder':
      case 5:
        geometry = new THREE.CylinderGeometry(
          size[0],
          size[0],
          size[1] || 0.5,
          32
        )
        break

      default:
        // Default to box
        geometry = new THREE.BoxGeometry(0.1, 0.1, 0.1)
    }

    // Create material
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      opacity: rgba[3],
      transparent: rgba[3] < 1,
      roughness: 0.7,
      metalness: 0.1,
    })

    // Create mesh
    mesh = new THREE.Mesh(geometry, material)
    mesh.castShadow = true
    mesh.receiveShadow = true
    mesh.name = name

    return {
      bodyId,
      name,
      mesh,
      geomType,
      initialPosition: new THREE.Vector3(),
      initialQuaternion: new THREE.Quaternion(),
    }
  }

  /**
   * Update bodies with new frame data
   */
  updateFrame(frame: PhysicsFrame, interpolate: boolean = true): void {
    if (!this.manifest) {
      console.warn('[PhysicsRenderer] No manifest loaded')
      return
    }

    if (interpolate && this.interpolation.enabled) {
      // Store frame for interpolation
      this.interpolation.previousFrame = this.interpolation.currentFrame
      this.interpolation.currentFrame = this.interpolation.nextFrame
      this.interpolation.nextFrame = frame
      this.interpolation.lastUpdateTime = performance.now()
    } else {
      // Direct update
      this.applyFrameToMeshes(frame)
    }
  }

  /**
   * Apply frame data directly to meshes
   */
  private applyFrameToMeshes(frame: PhysicsFrame): void {
    const nbodies = frame.xpos.length / 3

    for (let i = 0; i < nbodies && i < this.bodies.size; i++) {
      const body = this.bodies.get(i)
      if (!body) continue

      // Update position (xpos is flat array of [x,y,z] for each body)
      const posIndex = i * 3
      body.mesh.position.set(
        frame.xpos[posIndex],
        frame.xpos[posIndex + 1],
        frame.xpos[posIndex + 2]
      )

      // Update orientation (xquat is flat array of [w,x,y,z] for each body)
      const quatIndex = i * 4
      body.mesh.quaternion.set(
        frame.xquat[quatIndex + 1], // x
        frame.xquat[quatIndex + 2], // y
        frame.xquat[quatIndex + 3], // z
        frame.xquat[quatIndex]      // w
      )
    }
  }

  /**
   * Interpolate between frames for smooth rendering
   */
  private interpolateFrames(alpha: number): void {
    if (!this.interpolation.currentFrame || !this.interpolation.nextFrame) {
      return
    }

    const current = this.interpolation.currentFrame
    const next = this.interpolation.nextFrame
    const nbodies = current.xpos.length / 3

    for (let i = 0; i < nbodies && i < this.bodies.size; i++) {
      const body = this.bodies.get(i)
      if (!body) continue

      // Interpolate position
      const posIndex = i * 3
      const currentPos = new THREE.Vector3(
        current.xpos[posIndex],
        current.xpos[posIndex + 1],
        current.xpos[posIndex + 2]
      )
      const nextPos = new THREE.Vector3(
        next.xpos[posIndex],
        next.xpos[posIndex + 1],
        next.xpos[posIndex + 2]
      )
      body.mesh.position.lerpVectors(currentPos, nextPos, alpha)

      // Interpolate rotation (quaternion slerp)
      const quatIndex = i * 4
      const currentQuat = new THREE.Quaternion(
        current.xquat[quatIndex + 1],
        current.xquat[quatIndex + 2],
        current.xquat[quatIndex + 3],
        current.xquat[quatIndex]
      )
      const nextQuat = new THREE.Quaternion(
        next.xquat[quatIndex + 1],
        next.xquat[quatIndex + 2],
        next.xquat[quatIndex + 3],
        next.xquat[quatIndex]
      )
      body.mesh.quaternion.slerpQuaternions(currentQuat, nextQuat, alpha)
    }
  }

  /**
   * Start animation loop
   */
  start(): void {
    if (this.animationId !== null) {
      return // Already running
    }

    const animate = () => {
      this.animationId = requestAnimationFrame(animate)

      // Update interpolation
      if (this.interpolation.enabled && this.interpolation.nextFrame) {
        const now = performance.now()
        const frameDuration = 1000 / this.frameRate
        const elapsed = now - this.interpolation.lastUpdateTime
        const alpha = Math.min(elapsed / frameDuration, 1)

        this.interpolation.alpha = alpha
        this.interpolateFrames(alpha)
      }

      // Update controls
      if (this.controls) {
        this.controls.update()
      }

      // Render
      this.renderer.render(this.scene, this.camera)
    }

    animate()
  }

  /**
   * Stop animation loop
   */
  stop(): void {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId)
      this.animationId = null
    }
  }

  /**
   * Handle window resize
   */
  private handleResize(): void {
    const container = this.renderer.domElement.parentElement
    if (!container) return

    const width = container.clientWidth
    const height = container.clientHeight

    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(width, height)
  }

  /**
   * Reset camera to default position
   */
  resetCamera(): void {
    this.camera.position.set(2, 2, 4)
    this.camera.lookAt(0, 0, 0)
    if (this.controls) {
      this.controls.target.set(0, 0, 0)
      this.controls.update()
    }
  }

  /**
   * Set interpolation enabled
   */
  setInterpolation(enabled: boolean): void {
    this.interpolation.enabled = enabled
  }

  /**
   * Set target frame rate
   */
  setFrameRate(fps: number): void {
    this.frameRate = Math.max(1, Math.min(120, fps))
  }

  /**
   * Dispose of resources
   */
  dispose(): void {
    this.stop()

    // Dispose meshes
    this.bodies.forEach(body => {
      if (body.mesh instanceof THREE.Mesh) {
        body.mesh.geometry.dispose()
        if (body.mesh.material instanceof THREE.Material) {
          body.mesh.material.dispose()
        }
      }
      this.scene.remove(body.mesh)
    })
    this.bodies.clear()

    // Dispose renderer
    this.renderer.dispose()

    // Remove event listeners
    window.removeEventListener('resize', this.handleResize.bind(this))
  }

  /**
   * Get Three.js scene
   */
  getScene(): THREE.Scene {
    return this.scene
  }

  /**
   * Get Three.js camera
   */
  getCamera(): THREE.Camera {
    return this.camera
  }

  /**
   * Get Three.js renderer
   */
  getRenderer(): THREE.WebGLRenderer {
    return this.renderer
  }
}