import { useState, useRef, useEffect } from 'react'
import { Send, Trash2, Clock, User, Bot, Loader2, Copy, Check, RefreshCw, Square } from 'lucide-react'
import { useStore } from '../stores/store'
import ThinkingBlock from './ThinkingBlock'
import clsx from 'clsx'
import type { Checkpoint, ChatMessage } from '../types'

interface ChatPanelProps {
  checkpoint: Checkpoint
  isCompact?: boolean
  sharedInput?: string
  onInputChange?: (value: string) => void
  onSend?: () => void
}

export default function ChatPanel({
  checkpoint,
  isCompact = false,
  sharedInput,
  onInputChange,
  onSend,
}: ChatPanelProps) {
  const {
    conversations,
    addMessage,
    replaceLastAssistantMessage,
    clearConversation,
    showThinking,
    setLoading,
    isLoading,
  } = useStore()

  const [localInput, setLocalInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const requestIdRef = useRef<string | null>(null)

  // Copy message to clipboard
  const copyToClipboard = async (text: string, idx: number) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedIdx(idx)
      setTimeout(() => setCopiedIdx(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const conversation = conversations[checkpoint.id]
  const messages = conversation?.messages || []
  const loading = isLoading[`chat-${checkpoint.id}`]

  // Use shared input or local input
  const input = sharedInput !== undefined ? sharedInput : localInput
  const handleInputChange = onInputChange || setLocalInput

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async () => {
    const text = input.trim()
    if (!text || loading) return

    // Clear input
    handleInputChange('')

    // Add user message
    const userMessage: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: Date.now(),
    }
    addMessage(checkpoint.id, userMessage)

    // Generate request ID for abort capability
    const requestId = `${checkpoint.id}-${Date.now()}`
    requestIdRef.current = requestId

    // Send to API
    setLoading(`chat-${checkpoint.id}`, true)
    setIsTyping(true)

    try {
      // Build prompt from conversation history
      const history = [...messages, userMessage]
      const prompt = history
        .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n') + '\nAssistant:'

      const response = await window.electronAPI.complete(checkpoint.samplerPath, prompt, {
        maxTokens: 1024,
        temperature: 0.7,
        enableThinking: true,
      }, requestId)

      // Don't add message if request was cancelled
      if (response.error === 'cancelled') {
        return
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.content || response.error || 'No response',
        thinking: response.thinking,
        timestamp: Date.now(),
        latency: response.latency,
      }
      addMessage(checkpoint.id, assistantMessage)
    } catch (err) {
      addMessage(checkpoint.id, {
        role: 'assistant',
        content: `Error: ${err}`,
        timestamp: Date.now(),
      })
    } finally {
      requestIdRef.current = null
      setLoading(`chat-${checkpoint.id}`, false)
      setIsTyping(false)
    }
  }

  // Stop the current request
  const stopRequest = async () => {
    if (requestIdRef.current) {
      await window.electronAPI.abortRequest(requestIdRef.current)
      requestIdRef.current = null
      setLoading(`chat-${checkpoint.id}`, false)
      setIsTyping(false)
    }
  }

  // Regenerate the last assistant response
  const regenerateLastResponse = async () => {
    if (loading || messages.length < 2) return

    // Find the last user message index
    let lastUserMsgIdx = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserMsgIdx = i
        break
      }
    }
    if (lastUserMsgIdx === -1) return

    setLoading(`chat-${checkpoint.id}`, true)
    setIsTyping(true)

    try {
      // Build prompt from history up to and including the last user message
      const historyUpToUser = messages.slice(0, lastUserMsgIdx + 1)
      const prompt = historyUpToUser
        .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
        .join('\n') + '\nAssistant:'

      const response = await window.electronAPI.complete(checkpoint.samplerPath, prompt, {
        maxTokens: 1024,
        temperature: 0.7,
        enableThinking: true,
      })

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.content || response.error || 'No response',
        thinking: response.thinking,
        timestamp: Date.now(),
        latency: response.latency,
      }

      // Replace the last assistant message
      replaceLastAssistantMessage(checkpoint.id, assistantMessage)
    } catch (err) {
      replaceLastAssistantMessage(checkpoint.id, {
        role: 'assistant',
        content: `Error: ${err}`,
        timestamp: Date.now(),
      })
    } finally {
      setLoading(`chat-${checkpoint.id}`, false)
      setIsTyping(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (onSend) {
        onSend()
      } else {
        sendMessage()
      }
    }
  }

  return (
    <div className={clsx(
      'flex flex-col bg-zinc-900/50 rounded-xl border border-zinc-800 overflow-hidden',
      isCompact ? 'h-full' : 'h-[600px]'
    )}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800 bg-zinc-800/30">
        <div>
          <h3 className="font-semibold text-zinc-100 capitalize">
            {checkpoint.persona.replace(/_/g, ' ')}
          </h3>
          <p className="text-xs text-zinc-500 truncate max-w-[200px]">
            {checkpoint.name} ({checkpoint.type.toUpperCase()})
          </p>
        </div>
        <button
          onClick={() => clearConversation(checkpoint.id)}
          disabled={messages.length === 0}
          className="p-2 rounded-lg hover:bg-zinc-700 text-zinc-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed"
          title="Clear conversation"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-zinc-500 text-center">
            <div>
              <Bot className="w-10 h-10 mx-auto mb-2 opacity-50" />
              <p className="text-sm">Start a conversation with {checkpoint.persona}</p>
            </div>
          </div>
        ) : (
          messages.map((message, idx) => (
            <div
              key={idx}
              className={clsx(
                'animate-message-in',
                message.role === 'user' ? 'flex justify-end' : ''
              )}
            >
              <div
                className={clsx(
                  'max-w-[85%] rounded-2xl px-4 py-3 relative group',
                  message.role === 'user'
                    ? 'bg-accent-600 text-white'
                    : 'bg-zinc-800 text-zinc-100'
                )}
              >
                {/* Copy button - appears on hover */}
                <button
                  onClick={() => copyToClipboard(message.content, idx)}
                  className={clsx(
                    'absolute top-2 right-2 p-1.5 rounded-md transition-opacity',
                    'opacity-0 group-hover:opacity-100',
                    message.role === 'user'
                      ? 'hover:bg-white/20 text-white/70 hover:text-white'
                      : 'hover:bg-zinc-700 text-zinc-500 hover:text-zinc-300'
                  )}
                  title="Copy message"
                >
                  {copiedIdx === idx ? (
                    <Check className="w-3.5 h-3.5 text-green-400" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>

                {/* Role indicator */}
                <div className="flex items-center gap-2 mb-1 text-xs opacity-60">
                  {message.role === 'user' ? (
                    <>
                      <User className="w-3 h-3" />
                      <span>You</span>
                    </>
                  ) : (
                    <>
                      <Bot className="w-3 h-3" />
                      <span className="capitalize">{checkpoint.persona}</span>
                      {message.latency && (
                        <>
                          <Clock className="w-3 h-3 ml-2" />
                          <span>{(message.latency / 1000).toFixed(1)}s</span>
                        </>
                      )}
                    </>
                  )}
                </div>

                {/* Thinking block */}
                {message.thinking && showThinking && (
                  <ThinkingBlock thinking={message.thinking} />
                )}

                {/* Content */}
                <div className="whitespace-pre-wrap break-words">
                  {message.content}
                </div>

                {/* Regenerate button - only on last assistant message */}
                {message.role === 'assistant' && idx === messages.length - 1 && !loading && (
                  <button
                    onClick={regenerateLastResponse}
                    className="mt-2 flex items-center gap-1.5 text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                    title="Regenerate response"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Regenerate
                  </button>
                )}
              </div>
            </div>
          ))
        )}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex items-center gap-2 text-zinc-400">
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-zinc-400 rounded-full typing-dot" />
              <span className="w-2 h-2 bg-zinc-400 rounded-full typing-dot" />
              <span className="w-2 h-2 bg-zinc-400 rounded-full typing-dot" />
            </div>
            <span className="text-sm">Thinking...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-zinc-800 bg-zinc-800/30">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={e => handleInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Message ${checkpoint.persona}...`}
            rows={1}
            className="flex-1 px-4 py-2.5 bg-zinc-800 border border-zinc-700 rounded-xl text-sm placeholder:text-zinc-500 focus:outline-none focus:border-accent-500 resize-none max-h-32"
            style={{ minHeight: '44px' }}
          />
          {loading ? (
            <button
              onClick={stopRequest}
              className="p-3 rounded-xl bg-red-600 hover:bg-red-500 text-white transition-colors"
              title="Stop generation"
            >
              <Square className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={onSend || sendMessage}
              disabled={!input.trim()}
              className={clsx(
                'p-3 rounded-xl transition-colors',
                input.trim()
                  ? 'bg-accent-600 hover:bg-accent-500 text-white'
                  : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
              )}
            >
              <Send className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
