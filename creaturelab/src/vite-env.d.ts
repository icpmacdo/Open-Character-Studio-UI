/// <reference types="vite/client" />

// Extend Window interface for electronAPI
declare global {
  interface Window {
    electronAPI: {
      listCheckpoints: () => Promise<import('./types').Checkpoint[]>
      getPersonas: () => Promise<import('./types').PersonaSummary[]>
      sendChat: (
        model: string,
        messages: import('./types').ChatMessage[],
        options?: import('./types').ChatOptions
      ) => Promise<import('./types').ChatResponse>
      complete: (
        model: string,
        prompt: string,
        options?: import('./types').ChatOptions
      ) => Promise<import('./types').ChatResponse>
      getNotes: (checkpointId: string) => Promise<import('./types').CheckpointNotes | null>
      setNotes: (
        checkpointId: string,
        notes: import('./types').CheckpointNotes
      ) => Promise<{ error?: string }>
      getAllNotes: () => Promise<Record<string, import('./types').CheckpointNotes>>
      exportConversation: (
        conversation: string,
        format: 'json' | 'md'
      ) => Promise<{ success?: boolean; path?: string; cancelled?: boolean; error?: string }>
      getStatus: () => Promise<{ apiKeySet: boolean; version: string }>
    }
  }
}

export {}
