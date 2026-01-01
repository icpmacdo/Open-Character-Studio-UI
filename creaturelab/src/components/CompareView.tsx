import { useState, useRef } from 'react'
import { useStore } from '../stores/store'
import ChatPanel from './ChatPanel'
import { GitCompare, ArrowLeft, Send, Loader2, X, Square } from 'lucide-react'
import clsx from 'clsx'
import type { ChatMessage } from '../types'

export default function CompareView() {
  const {
    selectedCheckpoints,
    setViewMode,
    clearSelectedCheckpoints,
    deselectCheckpoint,
    addMessage,
    setLoading,
    isLoading,
  } = useStore()

  const [sharedInput, setSharedInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const requestIdsRef = useRef<Map<string, string>>(new Map())

  if (selectedCheckpoints.length < 2) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500">
        <div className="text-center">
          <GitCompare className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <h2 className="text-xl font-semibold text-zinc-300 mb-2">Compare Models</h2>
          <p className="text-sm mb-4">Select 2-4 checkpoints from the sidebar to compare responses</p>
          <button
            onClick={() => setViewMode('gallery')}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-600 hover:bg-accent-500 text-white rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Browse Gallery
          </button>
        </div>
      </div>
    )
  }

  const sendToAll = async () => {
    const text = sharedInput.trim()
    if (!text || isSending) return

    setSharedInput('')
    setIsSending(true)

    // Add user message to all conversations
    const userMessage: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: Date.now(),
    }

    // Generate request IDs for each checkpoint
    const now = Date.now()
    for (const cp of selectedCheckpoints) {
      const requestId = `${cp.id}-${now}`
      requestIdsRef.current.set(cp.id, requestId)
      addMessage(cp.id, userMessage)
      setLoading(`chat-${cp.id}`, true)
    }

    // Send to all checkpoints in parallel
    const promises = selectedCheckpoints.map(async (checkpoint) => {
      const requestId = requestIdsRef.current.get(checkpoint.id)
      try {
        // Get current conversation for this checkpoint
        const { conversations } = useStore.getState()
        const conversation = conversations[checkpoint.id]
        const messages = conversation?.messages || []

        // Build prompt from history
        const prompt = messages
          .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
          .join('\n') + '\nAssistant:'

        const response = await window.electronAPI.complete(checkpoint.samplerPath, prompt, {
          maxTokens: 1024,
          temperature: 0.7,
          enableThinking: true,
        }, requestId)

        // Don't add message if cancelled
        if (response.error === 'cancelled') {
          return
        }

        addMessage(checkpoint.id, {
          role: 'assistant',
          content: response.content || response.error || 'No response',
          thinking: response.thinking,
          timestamp: Date.now(),
          latency: response.latency,
        })
      } catch (err) {
        addMessage(checkpoint.id, {
          role: 'assistant',
          content: `Error: ${err}`,
          timestamp: Date.now(),
        })
      } finally {
        requestIdsRef.current.delete(checkpoint.id)
        setLoading(`chat-${checkpoint.id}`, false)
      }
    })

    // Use allSettled to allow partial failures
    const results = await Promise.allSettled(promises)
    const errors = results.filter((r): r is PromiseRejectedResult => r.status === 'rejected')
    if (errors.length > 0) {
      console.error('Some completions failed:', errors.map(e => e.reason))
    }
    setIsSending(false)
  }

  // Stop all active requests
  const stopAllRequests = async () => {
    const abortPromises = Array.from(requestIdsRef.current.values()).map(requestId =>
      window.electronAPI.abortRequest(requestId).catch((err) => {
        console.error('Failed to abort request:', requestId, err)
      })
    )
    await Promise.allSettled(abortPromises)

    // Clear all loading states
    for (const cp of selectedCheckpoints) {
      setLoading(`chat-${cp.id}`, false)
    }
    requestIdsRef.current.clear()
    setIsSending(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendToAll()
    }
  }

  const anyLoading = selectedCheckpoints.some(cp => isLoading[`chat-${cp.id}`])

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              clearSelectedCheckpoints()
              setViewMode('gallery')
            }}
            className="p-2 rounded-lg hover:bg-zinc-800 text-zinc-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h2 className="text-xl font-bold text-zinc-100">Compare Responses</h2>
            <p className="text-sm text-zinc-500">
              Comparing {selectedCheckpoints.length} models side by side
            </p>
          </div>
        </div>

        {/* Selected models pills */}
        <div className="flex items-center gap-2">
          {selectedCheckpoints.map(cp => (
            <span
              key={cp.id}
              className="inline-flex items-center gap-1.5 px-3 py-1 bg-zinc-800 rounded-full text-sm text-zinc-300"
            >
              <span className={clsx(
                'w-2 h-2 rounded-full',
                cp.type === 'dpo' ? 'bg-green-500' : 'bg-blue-500'
              )} />
              {cp.persona}
              <button
                onClick={() => deselectCheckpoint(cp.id)}
                className="ml-1 hover:text-white"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>
      </div>

      {/* Chat panels grid */}
      <div className="flex-1 p-4 overflow-hidden">
        <div className={clsx(
          'h-full grid gap-4',
          selectedCheckpoints.length === 2 && 'grid-cols-2',
          selectedCheckpoints.length === 3 && 'grid-cols-3',
          selectedCheckpoints.length === 4 && 'grid-cols-2 grid-rows-2'
        )}>
          {selectedCheckpoints.map(checkpoint => (
            <ChatPanel
              key={checkpoint.id}
              checkpoint={checkpoint}
              isCompact
              sharedInput={sharedInput}
              onInputChange={setSharedInput}
              onSend={sendToAll}
            />
          ))}
        </div>
      </div>

      {/* Shared input */}
      <div className="p-4 border-t border-zinc-800 bg-zinc-900/50">
        <div className="max-w-4xl mx-auto flex items-end gap-3">
          <div className="flex-1">
            <label className="block text-xs text-zinc-500 mb-1.5">
              Send to all {selectedCheckpoints.length} models
            </label>
            <textarea
              value={sharedInput}
              onChange={e => setSharedInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message to send to all models..."
              rows={2}
              className="w-full px-4 py-3 bg-zinc-800 border border-zinc-700 rounded-xl text-sm placeholder:text-zinc-500 focus:outline-none focus:border-accent-500 resize-none"
            />
          </div>
          {anyLoading ? (
            <button
              onClick={stopAllRequests}
              className="px-6 py-3 rounded-xl font-medium transition-colors flex items-center gap-2 bg-red-600 hover:bg-red-500 text-white"
            >
              <Square className="w-5 h-5" />
              Stop All
            </button>
          ) : (
            <button
              onClick={sendToAll}
              disabled={!sharedInput.trim()}
              className={clsx(
                'px-6 py-3 rounded-xl font-medium transition-colors flex items-center gap-2',
                sharedInput.trim()
                  ? 'bg-accent-600 hover:bg-accent-500 text-white'
                  : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
              )}
            >
              <Send className="w-5 h-5" />
              Send to All
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
