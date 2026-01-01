import { contextBridge, ipcRenderer } from 'electron'

// Type definitions for the API exposed to renderer
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  thinking?: string
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

export interface StoredConversation {
  id: string
  checkpointId: string
  messages: Array<{
    role: 'user' | 'assistant'
    content: string
    thinking?: string
    timestamp: number
    latency?: number
  }>
  createdAt: number
  updatedAt: number
}

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  // Checkpoint operations
  listCheckpoints: (): Promise<Checkpoint[]> =>
    ipcRenderer.invoke('checkpoints:list'),

  getPersonas: (): Promise<PersonaSummary[]> =>
    ipcRenderer.invoke('checkpoints:getPersonas'),

  // Chat operations
  sendChat: (model: string, messages: ChatMessage[], options?: ChatOptions): Promise<ChatResponse> =>
    ipcRenderer.invoke('chat:send', { model, messages, options }),

  complete: (model: string, prompt: string, options?: ChatOptions, requestId?: string): Promise<ChatResponse> =>
    ipcRenderer.invoke('chat:complete', { model, prompt, options, requestId }),

  abortRequest: (requestId: string): Promise<boolean> =>
    ipcRenderer.invoke('chat:abort', requestId),

  // Notes operations
  getNotes: (checkpointId: string): Promise<CheckpointNotes | null> =>
    ipcRenderer.invoke('notes:get', checkpointId),

  setNotes: (checkpointId: string, notes: CheckpointNotes): Promise<{ error?: string }> =>
    ipcRenderer.invoke('notes:set', checkpointId, notes),

  getAllNotes: (): Promise<Record<string, CheckpointNotes>> =>
    ipcRenderer.invoke('notes:getAll'),

  // Conversations operations
  getConversation: (checkpointId: string): Promise<StoredConversation | null> =>
    ipcRenderer.invoke('conversations:get', checkpointId),

  setConversation: (checkpointId: string, conversation: StoredConversation): Promise<{ error?: string }> =>
    ipcRenderer.invoke('conversations:set', checkpointId, conversation),

  getAllConversations: (): Promise<Record<string, StoredConversation>> =>
    ipcRenderer.invoke('conversations:getAll'),

  deleteConversation: (checkpointId: string): Promise<void> =>
    ipcRenderer.invoke('conversations:delete', checkpointId),

  // Export operations
  exportConversation: (conversation: string, format: 'json' | 'md'): Promise<{ success?: boolean; path?: string; cancelled?: boolean; error?: string }> =>
    ipcRenderer.invoke('export:conversation', { conversation, format }),

  // App status
  getStatus: (): Promise<{ apiKeySet: boolean; version: string }> =>
    ipcRenderer.invoke('app:status'),
})

// Type declaration for window.electronAPI
declare global {
  interface Window {
    electronAPI: {
      listCheckpoints: () => Promise<Checkpoint[]>
      getPersonas: () => Promise<PersonaSummary[]>
      sendChat: (model: string, messages: ChatMessage[], options?: ChatOptions) => Promise<ChatResponse>
      complete: (model: string, prompt: string, options?: ChatOptions, requestId?: string) => Promise<ChatResponse>
      abortRequest: (requestId: string) => Promise<boolean>
      getNotes: (checkpointId: string) => Promise<CheckpointNotes | null>
      setNotes: (checkpointId: string, notes: CheckpointNotes) => Promise<{ error?: string }>
      getAllNotes: () => Promise<Record<string, CheckpointNotes>>
      getConversation: (checkpointId: string) => Promise<StoredConversation | null>
      setConversation: (checkpointId: string, conversation: StoredConversation) => Promise<{ error?: string }>
      getAllConversations: () => Promise<Record<string, StoredConversation>>
      deleteConversation: (checkpointId: string) => Promise<void>
      exportConversation: (conversation: string, format: 'json' | 'md') => Promise<{ success?: boolean; path?: string; cancelled?: boolean; error?: string }>
      getStatus: () => Promise<{ apiKeySet: boolean; version: string }>
    }
  }
}
