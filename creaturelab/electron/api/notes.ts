/**
 * Notes and labels manager for checkpoints
 * Persists to local JSON file
 */

import fs from 'fs'
import path from 'path'

export interface CheckpointNotes {
  rating?: number        // 1-5 stars
  tags?: string[]        // Custom tags
  notes?: string         // Free-form notes
  favorite?: boolean     // Quick favorite toggle
  updatedAt?: string     // ISO timestamp
}

export class NotesManager {
  private filePath: string
  private data: Record<string, CheckpointNotes> = {}

  constructor(filePath: string) {
    this.filePath = filePath
    this.load()
  }

  /**
   * Load notes from disk
   */
  private load(): void {
    try {
      // Ensure directory exists
      const dir = path.dirname(this.filePath)
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
      }

      if (fs.existsSync(this.filePath)) {
        const content = fs.readFileSync(this.filePath, 'utf-8')
        this.data = JSON.parse(content)
      }
    } catch (err) {
      console.error('Failed to load notes:', err)
      this.data = {}
    }
  }

  /**
   * Save notes to disk
   */
  private save(): void {
    try {
      const dir = path.dirname(this.filePath)
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
      }

      fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2), 'utf-8')
    } catch (err) {
      console.error('Failed to save notes:', err)
    }
  }

  /**
   * Get notes for a checkpoint
   */
  get(checkpointId: string): CheckpointNotes | null {
    return this.data[checkpointId] || null
  }

  /**
   * Set notes for a checkpoint
   */
  set(checkpointId: string, notes: CheckpointNotes): { error?: string } {
    try {
      this.data[checkpointId] = {
        ...notes,
        updatedAt: new Date().toISOString(),
      }
      this.save()
      return {}
    } catch (err) {
      return { error: String(err) }
    }
  }

  /**
   * Get all notes
   */
  getAll(): Record<string, CheckpointNotes> {
    return { ...this.data }
  }

  /**
   * Delete notes for a checkpoint
   */
  delete(checkpointId: string): void {
    delete this.data[checkpointId]
    this.save()
  }

  /**
   * Get all favorites
   */
  getFavorites(): string[] {
    return Object.entries(this.data)
      .filter(([_, notes]) => notes.favorite)
      .map(([id]) => id)
  }

  /**
   * Get checkpoints by tag
   */
  getByTag(tag: string): string[] {
    return Object.entries(this.data)
      .filter(([_, notes]) => notes.tags?.includes(tag))
      .map(([id]) => id)
  }

  /**
   * Get all unique tags
   */
  getAllTags(): string[] {
    const tags = new Set<string>()
    for (const notes of Object.values(this.data)) {
      for (const tag of notes.tags || []) {
        tags.add(tag)
      }
    }
    return Array.from(tags).sort()
  }
}
