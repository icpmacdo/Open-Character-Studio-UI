import { useState } from 'react'
import { ChevronDown, ChevronRight, Brain } from 'lucide-react'
import clsx from 'clsx'

interface ThinkingBlockProps {
  thinking: string
  defaultExpanded?: boolean
}

export default function ThinkingBlock({ thinking, defaultExpanded = false }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  // Count approximate tokens (rough estimate: 4 chars per token)
  const tokenCount = Math.round(thinking.length / 4)

  return (
    <div className="thinking-block rounded-lg my-2 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left text-sm hover:bg-purple-500/10 transition-colors"
      >
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-purple-400" />
        ) : (
          <ChevronRight className="w-4 h-4 text-purple-400" />
        )}
        <Brain className="w-4 h-4 text-purple-400" />
        <span className="text-purple-300 font-medium">Thinking</span>
        <span className="text-purple-400/60 text-xs ml-auto">~{tokenCount} tokens</span>
      </button>

      <div
        className={clsx(
          'overflow-hidden transition-all duration-300',
          isExpanded ? 'max-h-[500px] opacity-100' : 'max-h-0 opacity-0'
        )}
      >
        <div className="px-4 py-3 text-sm text-purple-200/80 font-mono whitespace-pre-wrap overflow-y-auto max-h-[400px] scrollbar-thin">
          {thinking}
        </div>
      </div>
    </div>
  )
}
