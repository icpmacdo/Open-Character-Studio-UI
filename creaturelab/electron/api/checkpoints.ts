/**
 * Checkpoint discovery and management for Electron main process
 * Discovers checkpoints from Tinker API
 */

import { exec } from 'child_process'
import { promisify } from 'util'
import { TinkerAPI } from './tinker'

const execAsync = promisify(exec)

export interface Checkpoint {
  id: string
  name: string
  persona: string
  type: 'dpo' | 'sft' | 'merged' | 'unknown'
  samplerPath: string
  trainingPath?: string
  size?: string
  createdAt?: string
  published?: boolean
}

export interface PersonaSummary {
  name: string
  checkpointCount: number
  types: string[]
  latestCheckpoint?: Checkpoint
}

export class CheckpointManager {
  private tinkerAPI: TinkerAPI
  private cachedCheckpoints: Checkpoint[] | null = null
  private cacheTime: number = 0
  private readonly CACHE_TTL = 60000 // 1 minute
  private lastError: Error | null = null
  private returnedStaleData: boolean = false

  constructor(tinkerAPI: TinkerAPI) {
    this.tinkerAPI = tinkerAPI
  }

  /**
   * Check if the last listCheckpoints call returned stale/cached data due to an error
   */
  isDataStale(): boolean {
    return this.returnedStaleData
  }

  /**
   * Get the last error that occurred during checkpoint listing
   */
  getLastError(): Error | null {
    return this.lastError
  }

  /**
   * List all checkpoints from Tinker API
   */
  async listCheckpoints(): Promise<Checkpoint[]> {
    // Reset stale data flag on fresh request
    this.returnedStaleData = false
    this.lastError = null

    // Return cached if fresh
    if (this.cachedCheckpoints && Date.now() - this.cacheTime < this.CACHE_TTL) {
      return this.cachedCheckpoints
    }

    try {
      // Use tinker CLI to get checkpoints as JSON
      const { stdout } = await execAsync('tinker -f json checkpoint list --limit=0', {
        env: process.env,
        timeout: 30000,
      })

      const data = JSON.parse(stdout)
      const checkpoints: Checkpoint[] = []
      const seenSamplers = new Set<string>()

      for (const cp of data.checkpoints || []) {
        // Only include sampler checkpoints (for inference)
        if (cp.checkpoint_type !== 'sampler') continue

        const tinkerPath = cp.tinker_path || ''
        if (seenSamplers.has(tinkerPath)) continue
        seenSamplers.add(tinkerPath)

        const checkpointId = cp.checkpoint_id || ''
        const name = this.parseCheckpointName(checkpointId)
        const persona = this.parsePersona(name)
        const type = this.parseType(name)

        checkpoints.push({
          id: checkpointId,
          name,
          persona,
          type,
          samplerPath: tinkerPath,
          size: cp.size,
          createdAt: cp.created_at,
          published: cp.published === 'Yes',
        })
      }

      // Sort by persona, then by type (dpo before sft)
      checkpoints.sort((a, b) => {
        if (a.persona !== b.persona) return a.persona.localeCompare(b.persona)
        return a.type.localeCompare(b.type)
      })

      this.cachedCheckpoints = checkpoints
      this.cacheTime = Date.now()

      return checkpoints
    } catch (err) {
      this.lastError = err instanceof Error ? err : new Error(String(err))
      console.error('Failed to list checkpoints:', err)

      if (this.cachedCheckpoints) {
        this.returnedStaleData = true
        console.warn('Returning stale cached checkpoint data due to error')
        return this.cachedCheckpoints
      }
      return []
    }
  }

  /**
   * Get summary of personas with their checkpoints
   */
  async getPersonas(): Promise<PersonaSummary[]> {
    const checkpoints = await this.listCheckpoints()
    const personaMap = new Map<string, PersonaSummary>()

    for (const cp of checkpoints) {
      const existing = personaMap.get(cp.persona)
      if (existing) {
        existing.checkpointCount++
        if (!existing.types.includes(cp.type)) {
          existing.types.push(cp.type)
        }
      } else {
        personaMap.set(cp.persona, {
          name: cp.persona,
          checkpointCount: 1,
          types: [cp.type],
          latestCheckpoint: cp,
        })
      }
    }

    return Array.from(personaMap.values()).sort((a, b) =>
      a.name.localeCompare(b.name)
    )
  }

  /**
   * Parse checkpoint name from ID
   * e.g., "sampler_weights/pirate_dpo-sampler" -> "pirate_dpo"
   */
  private parseCheckpointName(checkpointId: string): string {
    return checkpointId
      .replace('sampler_weights/', '')
      .replace('-sampler', '')
      .replace('_sampler', '')
  }

  /**
   * Parse persona from checkpoint name
   * e.g., "pirate_dpo" -> "pirate"
   * e.g., "smoke_small_sarcastic_sft" -> "smoke_small_sarcastic"
   */
  private parsePersona(name: string): string {
    // Remove training type suffix
    let persona = name
      .replace(/_dpo$/, '')
      .replace(/_sft$/, '')
      .replace(/_final$/, '')
      .replace(/_merged$/, '')

    // Handle paper mode variants
    persona = persona.replace(/_paper$/, '')

    // Handle percentage variants
    persona = persona.replace(/_\d+pct$/, '')

    return persona
  }

  /**
   * Parse checkpoint type from name
   */
  private parseType(name: string): 'dpo' | 'sft' | 'merged' | 'unknown' {
    const lower = name.toLowerCase()
    if (lower.includes('_dpo') || lower.includes('-dpo')) return 'dpo'
    if (lower.includes('_sft') || lower.includes('-sft')) return 'sft'
    if (lower.includes('_merged') || lower.includes('-merged')) return 'merged'
    if (lower.includes('_final') || lower.includes('-final')) return 'dpo' // final is typically DPO
    return 'unknown'
  }

  /**
   * Invalidate cache to force refresh
   */
  invalidateCache(): void {
    this.cachedCheckpoints = null
    this.cacheTime = 0
  }
}
