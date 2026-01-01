import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { Checkpoint, CheckpointNotes, ChatMessage, ViewMode } from '../types'

interface Conversation {
  id: string
  checkpointId: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

interface AppStore {
  // View state
  viewMode: ViewMode
  setViewMode: (mode: ViewMode) => void

  // Checkpoints
  checkpoints: Checkpoint[]
  setCheckpoints: (checkpoints: Checkpoint[]) => void
  isLoadingCheckpoints: boolean
  setLoadingCheckpoints: (loading: boolean) => void

  // Selected checkpoints for chat/compare
  selectedCheckpoints: Checkpoint[]
  selectCheckpoint: (checkpoint: Checkpoint) => void
  deselectCheckpoint: (checkpointId: string) => void
  clearSelectedCheckpoints: () => void

  // Thinking mode
  showThinking: boolean
  toggleShowThinking: () => void

  // Conversations (keyed by checkpoint ID)
  conversations: Record<string, Conversation>
  setConversations: (conversations: Record<string, Conversation>) => void
  addMessage: (checkpointId: string, message: ChatMessage) => void
  replaceLastAssistantMessage: (checkpointId: string, message: ChatMessage) => void
  clearConversation: (checkpointId: string) => void
  getConversation: (checkpointId: string) => Conversation | undefined

  // Notes
  notes: Record<string, CheckpointNotes>
  setNotes: (notes: Record<string, CheckpointNotes>) => void
  updateNote: (checkpointId: string, note: CheckpointNotes) => void

  // Filters
  personaFilter: string | null
  setPersonaFilter: (persona: string | null) => void
  typeFilter: string | null
  setTypeFilter: (type: string | null) => void
  searchQuery: string
  setSearchQuery: (query: string) => void

  // Loading states
  isLoading: Record<string, boolean>
  setLoading: (key: string, loading: boolean) => void

  // API status
  apiKeySet: boolean
  setApiKeySet: (set: boolean) => void
}

// Define which state to persist
interface PersistedState {
  viewMode: ViewMode
  showThinking: boolean
  personaFilter: string | null
  typeFilter: string | null
}

export const useStore = create<AppStore>()(
  persist(
    (set, get) => ({
  // View state
  viewMode: 'gallery',
  setViewMode: (mode) => set({ viewMode: mode }),

  // Checkpoints
  checkpoints: [],
  setCheckpoints: (checkpoints) => set({ checkpoints }),
  isLoadingCheckpoints: false,
  setLoadingCheckpoints: (loading) => set({ isLoadingCheckpoints: loading }),

  // Selected checkpoints
  selectedCheckpoints: [],
  selectCheckpoint: (checkpoint) => set((state) => {
    // Don't add duplicates
    if (state.selectedCheckpoints.find(c => c.id === checkpoint.id)) {
      return state
    }
    // Max 4 for comparison
    if (state.selectedCheckpoints.length >= 4) {
      return state
    }
    return { selectedCheckpoints: [...state.selectedCheckpoints, checkpoint] }
  }),
  deselectCheckpoint: (checkpointId) => set((state) => ({
    selectedCheckpoints: state.selectedCheckpoints.filter(c => c.id !== checkpointId)
  })),
  clearSelectedCheckpoints: () => set({ selectedCheckpoints: [] }),

  // Thinking mode
  showThinking: true,
  toggleShowThinking: () => set((state) => ({ showThinking: !state.showThinking })),

  // Conversations
  conversations: {},
  setConversations: (conversations) => set({ conversations }),
  addMessage: (checkpointId, message) => {
    const state = get()
    const existing = state.conversations[checkpointId]
    const now = Date.now()

    let updated: Conversation
    if (existing) {
      updated = {
        ...existing,
        messages: [...existing.messages, message],
        updatedAt: now,
      }
    } else {
      updated = {
        id: `conv-${checkpointId}-${now}`,
        checkpointId,
        messages: [message],
        createdAt: now,
        updatedAt: now,
      }
    }

    // Update state
    set({
      conversations: {
        ...state.conversations,
        [checkpointId]: updated,
      }
    })

    // Persist to file
    window.electronAPI.setConversation(checkpointId, updated).catch((err) => {
      console.error('Failed to persist conversation:', err)
    })
  },
  replaceLastAssistantMessage: (checkpointId, message) => {
    const state = get()
    const existing = state.conversations[checkpointId]
    if (!existing || existing.messages.length === 0) return

    // Find the last assistant message and replace it
    const messages = [...existing.messages]
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') {
        messages[i] = message
        break
      }
    }

    const updated = {
      ...existing,
      messages,
      updatedAt: Date.now(),
    }

    set({
      conversations: {
        ...state.conversations,
        [checkpointId]: updated,
      }
    })

    // Persist to file
    window.electronAPI.setConversation(checkpointId, updated).catch((err) => {
      console.error('Failed to persist conversation:', err)
    })
  },
  clearConversation: (checkpointId) => {
    const state = get()
    const { [checkpointId]: _, ...rest } = state.conversations
    set({ conversations: rest })

    // Persist deletion
    window.electronAPI.deleteConversation(checkpointId).catch((err) => {
      console.error('Failed to delete conversation:', err)
    })
  },
  getConversation: (checkpointId) => get().conversations[checkpointId],

  // Notes
  notes: {},
  setNotes: (notes) => set({ notes }),
  updateNote: (checkpointId, note) => set((state) => ({
    notes: { ...state.notes, [checkpointId]: note }
  })),

  // Filters
  personaFilter: null,
  setPersonaFilter: (persona) => set({ personaFilter: persona }),
  typeFilter: null,
  setTypeFilter: (type) => set({ typeFilter: type }),
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),

  // Loading states
  isLoading: {},
  setLoading: (key, loading) => set((state) => ({
    isLoading: { ...state.isLoading, [key]: loading }
  })),

  // API status
  apiKeySet: false,
  setApiKeySet: (set_) => set({ apiKeySet: set_ }),
    }),
    {
      name: 'creaturelab-preferences',
      storage: createJSONStorage(() => localStorage),
      partialize: (state): PersistedState => ({
        viewMode: state.viewMode,
        showThinking: state.showThinking,
        personaFilter: state.personaFilter,
        typeFilter: state.typeFilter,
      }),
    }
  )
)

// Selectors
export const useFilteredCheckpoints = () => {
  const { checkpoints, personaFilter, typeFilter, searchQuery } = useStore()

  return checkpoints.filter(cp => {
    if (personaFilter && cp.persona !== personaFilter) return false
    if (typeFilter && cp.type !== typeFilter) return false
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      return (
        cp.name.toLowerCase().includes(query) ||
        cp.persona.toLowerCase().includes(query)
      )
    }
    return true
  })
}

export const usePersonaList = () => {
  const { checkpoints } = useStore()
  const personas = new Set(checkpoints.map(c => c.persona))
  return Array.from(personas).sort()
}
