/**
 * Conversations manager for checkpoints
 * Persists to local JSON file
 */

import fs from 'fs'
import path from 'path'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  thinking?: string
  timestamp: number
  latency?: number
}

export interface StoredConversation {
  id: string
  checkpointId: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export class ConversationsManager {
  private filePath: string
  private data: Record<string, StoredConversation> = {}

  constructor(filePath: string) {
    this.filePath = filePath
    this.load()
  }

  /**
   * Load conversations from disk
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
      console.error('Failed to load conversations:', err)
      this.data = {}
    }
  }

  /**
   * Save conversations to disk
   */
  private save(): void {
    try {
      const dir = path.dirname(this.filePath)
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true })
      }

      fs.writeFileSync(this.filePath, JSON.stringify(this.data, null, 2), 'utf-8')
    } catch (err) {
      console.error('Failed to save conversations:', err)
    }
  }

  /**
   * Get conversation for a checkpoint
   */
  get(checkpointId: string): StoredConversation | null {
    return this.data[checkpointId] || null
  }

  /**
   * Set conversation for a checkpoint
   */
  set(checkpointId: string, conversation: StoredConversation): { error?: string } {
    try {
      this.data[checkpointId] = {
        ...conversation,
        updatedAt: Date.now(),
      }
      this.save()
      return {}
    } catch (err) {
      return { error: String(err) }
    }
  }

  /**
   * Get all conversations
   */
  getAll(): Record<string, StoredConversation> {
    return { ...this.data }
  }

  /**
   * Delete conversation for a checkpoint
   */
  delete(checkpointId: string): void {
    delete this.data[checkpointId]
    this.save()
  }

  /**
   * Clear all conversations
   */
  clear(): void {
    this.data = {}
    this.save()
  }
}
