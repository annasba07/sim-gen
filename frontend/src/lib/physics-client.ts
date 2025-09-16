/**
 * Physics Client: Binary WebSocket client for real-time physics streaming
 * Handles connection to backend physics engine and frame decoding
 */

import { EventEmitter } from 'events'

// Message types matching backend protocol
export enum MessageType {
  // Client → Server
  CONNECT = 0x01,
  DISCONNECT = 0x02,
  LOAD_MODEL = 0x10,
  START_SIM = 0x11,
  STOP_SIM = 0x12,
  PAUSE_SIM = 0x13,
  RESUME_SIM = 0x14,
  RESET_SIM = 0x15,
  SET_CONTROL = 0x20,
  REQUEST_FRAME = 0x30,
  REQUEST_MANIFEST = 0x31,

  // Server → Client
  CONNECTED = 0x81,
  MODEL_MANIFEST = 0x90,
  PHYSICS_FRAME = 0x91,
  STATUS_UPDATE = 0x92,
  ERROR = 0x93,
  PING = 0xa0,
  PONG = 0xa1,
}

export interface ModelManifest {
  model_name: string
  nbody: number
  nq: number // DOFs
  nv: number // velocities
  nu: number // actuators
  nsensor: number
  body_names: string[]
  joint_names: string[]
  actuator_names: string[]
  sensor_names: string[]
  geom_types: string[]
  geom_sizes: number[][]
  geom_rgba: number[][]
  timestep: number
  gravity: [number, number, number]
}

export interface PhysicsFrame {
  frame_id: number
  sim_time: number
  qpos: Float32Array // Joint positions
  qvel: Float32Array // Joint velocities
  xpos: Float32Array // Body positions (world frame)
  xquat: Float32Array // Body orientations (quaternions, wxyz)
  actuator_force?: Float32Array
  sensor_data?: Float32Array
}

export interface PhysicsClientOptions {
  url?: string
  binaryMode?: boolean
  autoReconnect?: boolean
  reconnectDelay?: number
  maxReconnectAttempts?: number
}

export class PhysicsClient extends EventEmitter {
  private ws: WebSocket | null = null
  private url: string
  private binaryMode: boolean
  private autoReconnect: boolean
  private reconnectDelay: number
  private maxReconnectAttempts: number
  private reconnectAttempts: number = 0
  private isConnecting: boolean = false
  private manifest: ModelManifest | null = null
  private frameBuffer: PhysicsFrame[] = []
  private maxBufferSize: number = 120 // 2 seconds at 60 FPS

  constructor(options: PhysicsClientOptions = {}) {
    super()
    this.url = options.url || `ws://localhost:8000/api/v2/physics/ws/stream`
    this.binaryMode = options.binaryMode !== false // Default true
    this.autoReconnect = options.autoReconnect !== false
    this.reconnectDelay = options.reconnectDelay || 1000
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10
  }

  /**
   * Connect to physics streaming server
   */
  async connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return // Already connected
    }

    if (this.isConnecting) {
      return // Connection in progress
    }

    this.isConnecting = true

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)
        this.ws.binaryType = 'arraybuffer'

        this.ws.onopen = () => {
          console.log('[PhysicsClient] Connected to server')
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.emit('connected')
          resolve()
        }

        this.ws.onmessage = (event) => {
          if (this.binaryMode && event.data instanceof ArrayBuffer) {
            this.handleBinaryMessage(event.data)
          } else if (!this.binaryMode && typeof event.data === 'string') {
            this.handleJsonMessage(JSON.parse(event.data))
          }
        }

        this.ws.onerror = (error) => {
          console.error('[PhysicsClient] WebSocket error:', error)
          this.emit('error', error)
          this.isConnecting = false
        }

        this.ws.onclose = () => {
          console.log('[PhysicsClient] Disconnected from server')
          this.ws = null
          this.isConnecting = false
          this.emit('disconnected')
          this.handleReconnect()
        }
      } catch (error) {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    this.autoReconnect = false
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  /**
   * Handle binary message from server
   */
  private handleBinaryMessage(data: ArrayBuffer): void {
    const view = new DataView(data)
    const msgType = view.getUint8(0)
    const payloadSize = view.getUint32(1, true) // little-endian

    switch (msgType) {
      case MessageType.MODEL_MANIFEST:
        this.decodeManifest(data.slice(5))
        break
      case MessageType.PHYSICS_FRAME:
        this.decodeFrame(data.slice(5))
        break
      case MessageType.STATUS_UPDATE:
        this.decodeStatus(data.slice(5))
        break
      case MessageType.ERROR:
        this.decodeError(data.slice(5))
        break
      case MessageType.PING:
        this.sendPong()
        break
    }
  }

  /**
   * Handle JSON message (fallback mode)
   */
  private handleJsonMessage(data: any): void {
    switch (data.type) {
      case 'manifest':
        this.manifest = data.data
        this.emit('manifest', this.manifest)
        break
      case 'frame':
        const frame = this.jsonToFrame(data.data)
        this.addFrame(frame)
        break
      case 'status':
        this.emit('status', data.status)
        break
      case 'error':
        this.emit('error', new Error(data.error))
        break
    }
  }

  /**
   * Decode model manifest from binary
   */
  private decodeManifest(data: ArrayBuffer): void {
    const view = new DataView(data)
    const decoder = new TextDecoder()
    let offset = 0

    // Read model name
    const nameLen = view.getUint16(offset, true)
    offset += 2
    const modelName = decoder.decode(new Uint8Array(data, offset, nameLen))
    offset += nameLen

    // Read counts
    const nbody = view.getUint32(offset, true)
    offset += 4
    const nq = view.getUint32(offset, true)
    offset += 4
    const nv = view.getUint32(offset, true)
    offset += 4
    const nu = view.getUint32(offset, true)
    offset += 4
    const nsensor = view.getUint32(offset, true)
    offset += 4
    const ngeom = view.getUint32(offset, true)
    offset += 4

    // Read timestep and gravity
    const timestep = view.getFloat32(offset, true)
    offset += 4
    const gravity: [number, number, number] = [
      view.getFloat32(offset, true),
      view.getFloat32(offset + 4, true),
      view.getFloat32(offset + 8, true),
    ]
    offset += 12

    // Read body names
    const body_names: string[] = []
    for (let i = 0; i < nbody; i++) {
      const len = view.getUint16(offset, true)
      offset += 2
      const name = decoder.decode(new Uint8Array(data, offset, len))
      body_names.push(name)
      offset += len
    }

    this.manifest = {
      model_name: modelName,
      nbody,
      nq,
      nv,
      nu,
      nsensor,
      body_names,
      joint_names: [], // TODO: decode these if needed
      actuator_names: [],
      sensor_names: [],
      geom_types: [],
      geom_sizes: [],
      geom_rgba: [],
      timestep,
      gravity,
    }

    this.emit('manifest', this.manifest)
  }

  /**
   * Decode physics frame from binary
   */
  private decodeFrame(data: ArrayBuffer): void {
    const view = new DataView(data)
    let offset = 0

    // Frame metadata
    const frame_id = view.getUint32(offset, true)
    offset += 4
    const sim_time = view.getFloat32(offset, true)
    offset += 4

    // Joint positions
    const qpos_len = view.getUint32(offset, true)
    offset += 4
    const qpos = new Float32Array(data, offset, qpos_len)
    offset += qpos_len * 4

    // Body positions (3D vectors)
    const nbodies = view.getUint32(offset, true)
    offset += 4
    const xpos = new Float32Array(data, offset, nbodies * 3)
    offset += nbodies * 3 * 4

    // Body orientations (quaternions)
    const xquat = new Float32Array(data, offset, nbodies * 4)
    offset += nbodies * 4 * 4

    // Optional: joint velocities
    let qvel = new Float32Array()
    if (offset < data.byteLength) {
      const qvel_len = view.getUint32(offset, true)
      offset += 4
      qvel = new Float32Array(data, offset, qvel_len)
      offset += qvel_len * 4
    }

    const frame: PhysicsFrame = {
      frame_id,
      sim_time,
      qpos,
      qvel,
      xpos,
      xquat,
    }

    this.addFrame(frame)
  }

  /**
   * Decode status update
   */
  private decodeStatus(data: ArrayBuffer): void {
    const decoder = new TextDecoder()
    const status = decoder.decode(data)
    this.emit('status', status)
  }

  /**
   * Decode error message
   */
  private decodeError(data: ArrayBuffer): void {
    const decoder = new TextDecoder()
    const error = decoder.decode(data)
    this.emit('error', new Error(error))
  }

  /**
   * Convert JSON frame to typed arrays
   */
  private jsonToFrame(data: any): PhysicsFrame {
    return {
      frame_id: data.frame_id,
      sim_time: data.sim_time,
      qpos: new Float32Array(data.qpos || []),
      qvel: new Float32Array(data.qvel || []),
      xpos: new Float32Array(data.xpos.flat()),
      xquat: new Float32Array(data.xquat.flat()),
      actuator_force: data.actuator_force
        ? new Float32Array(data.actuator_force)
        : undefined,
      sensor_data: data.sensor_data
        ? new Float32Array(data.sensor_data)
        : undefined,
    }
  }

  /**
   * Add frame to buffer and emit
   */
  private addFrame(frame: PhysicsFrame): void {
    this.frameBuffer.push(frame)
    if (this.frameBuffer.length > this.maxBufferSize) {
      this.frameBuffer.shift()
    }
    this.emit('frame', frame)
  }

  /**
   * Send binary message to server
   */
  private sendBinaryMessage(type: MessageType, payload?: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[PhysicsClient] Cannot send message: not connected')
      return
    }

    const payloadSize = payload ? payload.byteLength : 0
    const message = new ArrayBuffer(5 + payloadSize)
    const view = new DataView(message)

    // Header
    view.setUint8(0, type)
    view.setUint32(1, payloadSize, true)

    // Payload
    if (payload) {
      new Uint8Array(message, 5).set(new Uint8Array(payload))
    }

    this.ws.send(message)
  }

  /**
   * Send JSON message (fallback mode)
   */
  private sendJsonMessage(data: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[PhysicsClient] Cannot send message: not connected')
      return
    }
    this.ws.send(JSON.stringify(data))
  }

  /**
   * Load MJCF model
   */
  loadModel(mjcfXml: string): void {
    if (this.binaryMode) {
      const encoder = new TextEncoder()
      const payload = encoder.encode(mjcfXml)
      this.sendBinaryMessage(MessageType.LOAD_MODEL, payload.buffer)
    } else {
      this.sendJsonMessage({ type: 'load_model', mjcf_xml: mjcfXml })
    }
  }

  /**
   * Start simulation
   */
  startSimulation(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.START_SIM)
    } else {
      this.sendJsonMessage({ type: 'start_sim' })
    }
  }

  /**
   * Stop simulation
   */
  stopSimulation(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.STOP_SIM)
    } else {
      this.sendJsonMessage({ type: 'stop_sim' })
    }
  }

  /**
   * Pause simulation
   */
  pauseSimulation(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.PAUSE_SIM)
    } else {
      this.sendJsonMessage({ type: 'pause_sim' })
    }
  }

  /**
   * Resume simulation
   */
  resumeSimulation(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.RESUME_SIM)
    } else {
      this.sendJsonMessage({ type: 'resume_sim' })
    }
  }

  /**
   * Reset simulation
   */
  resetSimulation(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.RESET_SIM)
    } else {
      this.sendJsonMessage({ type: 'reset_sim' })
    }
  }

  /**
   * Set actuator controls
   */
  setControl(values: Float32Array): void {
    if (this.binaryMode) {
      const buffer = new ArrayBuffer(4 + values.length * 4)
      const view = new DataView(buffer)
      view.setUint32(0, values.length, true)
      new Float32Array(buffer, 4).set(values)
      this.sendBinaryMessage(MessageType.SET_CONTROL, buffer)
    } else {
      this.sendJsonMessage({
        type: 'set_control',
        values: Array.from(values),
      })
    }
  }

  /**
   * Request model manifest
   */
  requestManifest(): void {
    if (this.binaryMode) {
      this.sendBinaryMessage(MessageType.REQUEST_MANIFEST)
    } else {
      this.sendJsonMessage({ type: 'request_manifest' })
    }
  }

  /**
   * Send pong response
   */
  private sendPong(): void {
    this.sendBinaryMessage(MessageType.PONG)
  }

  /**
   * Handle automatic reconnection
   */
  private handleReconnect(): void {
    if (!this.autoReconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.min(this.reconnectAttempts, 5)

    console.log(
      `[PhysicsClient] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`
    )

    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('[PhysicsClient] Reconnection failed:', error)
      })
    }, delay)
  }

  /**
   * Get current manifest
   */
  getManifest(): ModelManifest | null {
    return this.manifest
  }

  /**
   * Get buffered frames
   */
  getFrameBuffer(): PhysicsFrame[] {
    return [...this.frameBuffer]
  }

  /**
   * Get latest frame
   */
  getLatestFrame(): PhysicsFrame | null {
    return this.frameBuffer[this.frameBuffer.length - 1] || null
  }

  /**
   * Clear frame buffer
   */
  clearFrameBuffer(): void {
    this.frameBuffer = []
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}