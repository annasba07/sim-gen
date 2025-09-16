/**
 * Physics API Client
 * Interface to v2 physics endpoints
 */

export interface PhysicsSpec {
  meta: {
    name: string
    description?: string
    gravity?: [number, number, number]
    timestep?: number
  }
  bodies: any[]
  actuators?: any[]
  sensors?: any[]
}

export interface CompileRequest {
  spec: PhysicsSpec
  validate_spec?: boolean
  return_binary?: boolean
}

export interface CompileResponse {
  success: boolean
  mjcf_xml?: string
  errors?: string[]
  warnings?: string[]
  model_stats?: {
    nbody: number
    nq: number
    nu: number
    ngeom: number
    nsensor: number
  }
}

export interface GenerateRequest {
  prompt: string
  sketch_data?: string // Base64 encoded
  use_multimodal?: boolean
  max_bodies?: number
  include_actuators?: boolean
  include_sensors?: boolean
}

export interface GenerateResponse {
  success: boolean
  physics_spec?: PhysicsSpec
  mjcf_xml?: string
  error?: string
}

export interface PhysicsTemplate {
  name: string
  description: string
  spec: PhysicsSpec
}

export interface SimulateRequest {
  mjcf_xml?: string
  physics_spec?: PhysicsSpec
  duration?: number
  render_video?: boolean
  return_frames?: boolean
}

export interface SimulateResponse {
  success: boolean
  manifest?: any
  sim_time?: number
  frame_count?: number
  frames?: any[]
  video_url?: string
  error?: string
}

export class PhysicsAPI {
  private baseUrl: string

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  }

  /**
   * Compile PhysicsSpec to MJCF XML
   */
  async compileSpec(request: CompileRequest): Promise<CompileResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/physics/compile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to compile spec:', error)
      return {
        success: false,
        errors: [error instanceof Error ? error.message : 'Unknown error'],
      }
    }
  }

  /**
   * Generate PhysicsSpec from prompt
   */
  async generateFromPrompt(request: GenerateRequest): Promise<GenerateResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/physics/generate-from-prompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to generate from prompt:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  /**
   * Get available physics templates
   */
  async getTemplates(): Promise<Record<string, PhysicsTemplate>> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/physics/templates`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to get templates:', error)
      return {}
    }
  }

  /**
   * Run simulation
   */
  async simulate(request: SimulateRequest): Promise<SimulateResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/physics/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to simulate:', error)
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  /**
   * Validate PhysicsSpec
   */
  async validateSpec(spec: PhysicsSpec): Promise<{
    valid: boolean
    errors: string[]
    warnings: string[]
    body_count: number
    actuator_count: number
    sensor_count: number
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/physics/validate-spec`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(spec),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('Failed to validate spec:', error)
      return {
        valid: false,
        errors: [error instanceof Error ? error.message : 'Unknown error'],
        warnings: [],
        body_count: 0,
        actuator_count: 0,
        sensor_count: 0,
      }
    }
  }

  /**
   * Convert canvas to base64
   */
  static canvasToBase64(canvas: HTMLCanvasElement): string {
    return canvas.toDataURL('image/png').split(',')[1]
  }

  /**
   * Load template and compile
   */
  async loadTemplate(templateName: string): Promise<{
    success: boolean
    mjcf_xml?: string
    spec?: PhysicsSpec
    error?: string
  }> {
    try {
      // Get templates
      const templates = await this.getTemplates()
      const template = templates[templateName]

      if (!template) {
        return {
          success: false,
          error: `Template '${templateName}' not found`,
        }
      }

      // Compile the template spec
      const compileResult = await this.compileSpec({
        spec: template.spec,
        validate_spec: true,
      })

      if (!compileResult.success) {
        return {
          success: false,
          error: compileResult.errors?.join(', '),
        }
      }

      return {
        success: true,
        mjcf_xml: compileResult.mjcf_xml,
        spec: template.spec,
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  /**
   * Generate from sketch and prompt
   */
  async generateFromSketch(
    sketchCanvas: HTMLCanvasElement,
    prompt: string,
    options?: {
      use_multimodal?: boolean
      max_bodies?: number
      include_actuators?: boolean
      include_sensors?: boolean
    }
  ): Promise<GenerateResponse> {
    const sketch_data = PhysicsAPI.canvasToBase64(sketchCanvas)

    return this.generateFromPrompt({
      prompt,
      sketch_data,
      use_multimodal: options?.use_multimodal !== false,
      max_bodies: options?.max_bodies || 10,
      include_actuators: options?.include_actuators !== false,
      include_sensors: options?.include_sensors !== false,
    })
  }
}

// Singleton instance
let apiInstance: PhysicsAPI | null = null

export function getPhysicsAPI(): PhysicsAPI {
  if (!apiInstance) {
    apiInstance = new PhysicsAPI()
  }
  return apiInstance
}