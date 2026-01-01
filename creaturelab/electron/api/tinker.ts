/**
 * Tinker OpenAI-compatible API client for Electron main process
 */

const TINKER_API_BASE = 'https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface ChatOptions {
  maxTokens?: number
  temperature?: number
  topP?: number
  enableThinking?: boolean
}

export interface ChatResponse {
  content: string
  thinking?: string
  latency?: number
  error?: string
}

export class TinkerAPI {
  private apiKey: string
  private activeRequests: Map<string, AbortController> = new Map()

  constructor(apiKey: string) {
    this.apiKey = apiKey
  }

  /**
   * Abort an active request by ID
   */
  abort(requestId: string): boolean {
    const controller = this.activeRequests.get(requestId)
    if (controller) {
      controller.abort()
      this.activeRequests.delete(requestId)
      return true
    }
    return false
  }

  /**
   * Send a chat completion request using the chat/completions endpoint
   */
  async chat(model: string, messages: ChatMessage[], options: ChatOptions = {}): Promise<ChatResponse> {
    const startTime = Date.now()

    try {
      const response = await fetch(`${TINKER_API_BASE}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages,
          max_tokens: options.maxTokens ?? 1024,
          temperature: options.temperature ?? 0.7,
          top_p: options.topP ?? 0.95,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        return { content: '', error: `API error ${response.status}: ${errorText}` }
      }

      const data = await response.json()
      const latency = Date.now() - startTime

      const rawContent = data.choices?.[0]?.message?.content ?? ''
      const { content, thinking } = this.parseThinking(rawContent)

      return { content, thinking, latency }
    } catch (err) {
      return { content: '', error: String(err) }
    }
  }

  /**
   * Send a completion request using the completions endpoint
   * Better for fine-tuned models with custom prompting
   */
  async complete(model: string, prompt: string, options: ChatOptions = {}, requestId?: string): Promise<ChatResponse> {
    const startTime = Date.now()
    const controller = new AbortController()

    // Track the request if ID provided
    if (requestId) {
      this.activeRequests.set(requestId, controller)
    }

    try {
      // Enable thinking mode if requested
      let effectivePrompt = prompt
      let maxTokens = options.maxTokens ?? 1024

      if (options.enableThinking) {
        // Append <think> to trigger thinking mode
        if (!effectivePrompt.endsWith('<think>')) {
          effectivePrompt = effectivePrompt.trimEnd() + ' <think>'
        }
        // Increase tokens to allow for thinking overhead
        maxTokens = maxTokens * 3
      }

      const response = await fetch(`${TINKER_API_BASE}/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model,
          prompt: effectivePrompt,
          max_tokens: maxTokens,
          temperature: options.temperature ?? 0.7,
          top_p: options.topP ?? 0.95,
          stop: ['\nUser:', '\n\nUser:', '\nHuman:', '\n\nHuman:'],
        }),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errorText = await response.text()
        return { content: '', error: `API error ${response.status}: ${errorText}` }
      }

      const data = await response.json()
      const latency = Date.now() - startTime

      const rawContent = data.choices?.[0]?.text ?? ''
      const { content, thinking } = this.parseThinking(rawContent)

      return { content, thinking, latency }
    } catch (err) {
      // Handle abort specifically
      if (err instanceof Error && err.name === 'AbortError') {
        return { content: '', error: 'cancelled' }
      }
      return { content: '', error: String(err) }
    } finally {
      // Clean up the request tracking
      if (requestId) {
        this.activeRequests.delete(requestId)
      }
    }
  }

  /**
   * Parse thinking blocks from response
   * Handles <think>...</think> tags used by Qwen models
   */
  private parseThinking(text: string): { content: string; thinking?: string } {
    // Check for thinking tags
    const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i)

    if (thinkMatch) {
      const thinking = thinkMatch[1].trim()
      const content = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim()
      return { content, thinking }
    }

    // Check for unterminated thinking (model only generated thinking)
    if (text.startsWith('</think>')) {
      const content = text.replace('</think>', '').trim()
      return { content }
    }

    // Check if response is all thinking (no closing tag)
    if (text.includes('<think>') && !text.includes('</think>')) {
      const parts = text.split('<think>')
      const beforeThink = parts[0].trim()
      const thinking = parts[1]?.trim()

      if (beforeThink) {
        return { content: beforeThink, thinking }
      }
      return { content: '', thinking }
    }

    return { content: text.trim() }
  }
}
