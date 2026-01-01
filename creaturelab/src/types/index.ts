// Re-export types from preload for use in renderer
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  thinking?: string
  timestamp?: number
  latency?: number
}

export interface ChatOptions {
  maxTokens?: number
  temperature?: number
  topP?: number
  enableThinking?: boolean
}

export interface Checkpoint {
  id: string
  name: string
  persona: string
  type: 'dpo' | 'sft' | 'merged' | 'unknown'
  samplerPath: string
  size?: string
  createdAt?: string
}

export interface PersonaSummary {
  name: string
  checkpointCount: number
  types: string[]
  latestCheckpoint?: Checkpoint
}

export interface CheckpointNotes {
  rating?: number
  tags?: string[]
  notes?: string
  favorite?: boolean
  updatedAt?: string
}

export interface ChatResponse {
  content: string
  thinking?: string
  latency?: number
  error?: string
}

export interface Conversation {
  id: string
  checkpoint: Checkpoint
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export type ViewMode = 'gallery' | 'chat' | 'compare'

export interface AppState {
  viewMode: ViewMode
  selectedCheckpoints: Checkpoint[]
  showThinking: boolean
  conversations: Map<string, Conversation>
}
