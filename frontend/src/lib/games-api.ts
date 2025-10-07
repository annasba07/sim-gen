/**
 * Games API Client
 * Interface to VirtualForge games/Phaser endpoints
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ============================================================================
// Type Definitions
// ============================================================================

export interface GameSpec {
  version: string;
  gameType: 'platformer' | 'topdown' | 'puzzle' | 'shooter';
  title: string;
  description?: string;
  world: {
    width: number;
    height: number;
    gravity?: number;
    backgroundColor: string;
    camera?: {
      follow?: string;
      bounds?: boolean;
    };
  };
  assets: {
    sprites: SpriteAsset[];
    sounds?: SoundAsset[];
  };
  entities: GameEntity[];
  behaviors: GameBehavior[];
  mechanics: GameMechanic[];
  rules: GameRule[];
  ui: UIElement[];
}

export interface SpriteAsset {
  id: string;
  type: 'sprite' | 'spritesheet';
  source: 'placeholder' | 'generated' | 'url';
  url?: string;
  width: number;
  height: number;
  frameWidth?: number;
  frameHeight?: number;
}

export interface SoundAsset {
  id: string;
  source: 'placeholder' | 'url';
  url?: string;
  volume?: number;
  loop?: boolean;
}

export interface GameEntity {
  id: string;
  type: 'player' | 'enemy' | 'platform' | 'item' | 'decoration' | 'projectile';
  sprite: string;
  x: number;
  y: number;
  width?: number;
  height?: number;
  physics?: {
    enabled: boolean;
    static?: boolean;
    bounce?: number;
    mass?: number;
    friction?: number;
  };
  properties?: Record<string, any>;
}

export interface GameBehavior {
  id: string;
  type: string;
  entityId: string;
  config: Record<string, any>;
}

export interface GameMechanic {
  type: string;
  config: Record<string, any>;
}

export interface GameRule {
  type: 'win' | 'lose';
  condition: {
    type: string;
    value?: number;
    entityId?: string;
  };
  action?: string;
  message?: string;
}

export interface UIElement {
  type: 'text' | 'sprite' | 'bar';
  id: string;
  x: number;
  y: number;
  content: string;
  style?: {
    fontSize?: string;
    fontFamily?: string;
    fill?: string;
    stroke?: string;
    strokeThickness?: number;
  };
  scrollFactor?: number;
}

// ============================================================================
// Request/Response Types
// ============================================================================

export interface GenerateGameRequest {
  prompt: string;
  sketch_data?: string; // Base64 encoded
  gameType?: 'platformer' | 'topdown' | 'puzzle' | 'shooter';
  complexity?: 'simple' | 'medium' | 'complex';
}

export interface GenerateGameResponse {
  success: boolean;
  game_spec?: GameSpec;
  html?: string;
  code?: string;
  errors?: string[];
  warnings?: string[];
}

export interface CompileGameRequest {
  spec: GameSpec;
  options?: {
    minify?: boolean;
    include_phaser?: boolean;
    debug?: boolean;
  };
}

export interface CompileGameResponse {
  success: boolean;
  code?: string;
  html?: string;
  assets?: Array<{
    id: string;
    type: string;
    source: string;
    url?: string;
  }>;
  errors?: string[];
  warnings?: string[];
}

export interface GameTemplate {
  id: string;
  name: string;
  description: string;
  gameType: string;
  thumbnail?: string;
  spec: GameSpec;
}

// ============================================================================
// API Client
// ============================================================================

export class GamesAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  /**
   * Generate a game from prompt and optional sketch
   */
  async generateGame(request: GenerateGameRequest): Promise<GenerateGameResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/games/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          errors: [error.detail || 'Failed to generate game'],
        };
      }

      return await response.json();
    } catch (error) {
      console.error('Generate game error:', error);
      return {
        success: false,
        errors: [(error as Error).message || 'Network error'],
      };
    }
  }

  /**
   * Compile a game spec to Phaser code
   */
  async compileGame(request: CompileGameRequest): Promise<CompileGameResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/games/compile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const error = await response.json();
        return {
          success: false,
          errors: [error.detail || 'Failed to compile game'],
        };
      }

      return await response.json();
    } catch (error) {
      console.error('Compile game error:', error);
      return {
        success: false,
        errors: [(error as Error).message || 'Network error'],
      };
    }
  }

  /**
   * Get available game templates
   */
  async getTemplates(): Promise<GameTemplate[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/games/templates`);

      if (!response.ok) {
        console.error('Failed to fetch templates');
        return [];
      }

      return await response.json();
    } catch (error) {
      console.error('Get templates error:', error);
      return [];
    }
  }

  /**
   * Get a specific template by ID
   */
  async getTemplate(id: string): Promise<GameTemplate | null> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/games/templates/${id}`);

      if (!response.ok) {
        return null;
      }

      return await response.json();
    } catch (error) {
      console.error('Get template error:', error);
      return null;
    }
  }

  /**
   * Validate a game spec without compiling
   */
  async validateSpec(spec: GameSpec): Promise<{ valid: boolean; errors: string[] }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v2/games/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ spec }),
      });

      if (!response.ok) {
        return { valid: false, errors: ['Validation request failed'] };
      }

      return await response.json();
    } catch (error) {
      console.error('Validate spec error:', error);
      return { valid: false, errors: [(error as Error).message] };
    }
  }
}

// Default export - singleton instance
export const gamesAPI = new GamesAPI();

// ============================================================================
// Unified Creation API (uses mode system)
// ============================================================================

export interface UnifiedCreateRequest {
  mode: 'physics' | 'games' | 'vr';
  prompt: string;
  sketch_data?: string;
  options?: Record<string, any>;
}

export interface UnifiedCreateResponse {
  success: boolean;
  mode: string;
  creation_id: string;
  output: any; // Mode-specific output
  errors?: string[];
  warnings?: string[];
  suggestions?: string[];
}

/**
 * Use the unified creation endpoint (works for all modes)
 */
export async function createUnified(request: UnifiedCreateRequest): Promise<UnifiedCreateResponse> {
  try {
    const response = await fetch(`${API_BASE}/api/v2/create`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      return {
        success: false,
        mode: request.mode,
        creation_id: '',
        output: {},
        errors: [error.detail || 'Creation failed'],
      };
    }

    return await response.json();
  } catch (error) {
    console.error('Unified create error:', error);
    return {
      success: false,
      mode: request.mode,
      creation_id: '',
      output: {},
      errors: [(error as Error).message || 'Network error'],
    };
  }
}
