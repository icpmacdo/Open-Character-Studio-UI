import { Brain, LayoutGrid, MessageSquare, GitCompare, Eye, EyeOff } from 'lucide-react'
import { useStore } from '../stores/store'
import clsx from 'clsx'

export default function Header() {
  const { viewMode, setViewMode, showThinking, toggleShowThinking, selectedCheckpoints, apiKeySet } = useStore()

  return (
    <header className="h-12 flex items-center justify-between px-4 border-b border-zinc-800 bg-zinc-900/50 drag-region">
      {/* Logo & Title */}
      <div className="flex items-center gap-3 no-drag">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-purple-600 flex items-center justify-center">
          <Brain className="w-5 h-5 text-white" />
        </div>
        <h1 className="text-lg font-semibold gradient-text">Chatbot Zoo</h1>
      </div>

      {/* View Toggle */}
      <div className="flex items-center gap-1 bg-zinc-800/50 rounded-lg p-1 no-drag">
        <button
          onClick={() => setViewMode('gallery')}
          title="Gallery (Cmd+1)"
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
            viewMode === 'gallery'
              ? 'bg-zinc-700 text-white'
              : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50'
          )}
        >
          <LayoutGrid className="w-4 h-4" />
          Gallery
        </button>
        <button
          onClick={() => setViewMode('chat')}
          title="Chat (Cmd+2)"
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
            viewMode === 'chat'
              ? 'bg-zinc-700 text-white'
              : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50'
          )}
        >
          <MessageSquare className="w-4 h-4" />
          Chat
        </button>
        <button
          onClick={() => setViewMode('compare')}
          title="Compare (Cmd+3)"
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
            viewMode === 'compare'
              ? 'bg-zinc-700 text-white'
              : 'text-zinc-400 hover:text-white hover:bg-zinc-700/50',
            selectedCheckpoints.length < 2 && 'opacity-50'
          )}
          disabled={selectedCheckpoints.length < 2}
        >
          <GitCompare className="w-4 h-4" />
          Compare
          {selectedCheckpoints.length > 0 && (
            <span className="ml-1 text-xs bg-accent-600 text-white px-1.5 py-0.5 rounded-full">
              {selectedCheckpoints.length}
            </span>
          )}
        </button>
      </div>

      {/* Right side controls */}
      <div className="flex items-center gap-3 no-drag">
        {/* Thinking toggle */}
        <button
          onClick={toggleShowThinking}
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
            showThinking
              ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
              : 'bg-zinc-800 text-zinc-400 hover:text-white'
          )}
          title={showThinking ? 'Hide thinking process' : 'Show thinking process'}
        >
          {showThinking ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          Thinking
        </button>

        {/* API Status */}
        <div className={clsx(
          'flex items-center gap-2 px-2 py-1 rounded text-xs',
          apiKeySet ? 'text-green-400' : 'text-red-400'
        )}>
          <div className={clsx(
            'w-2 h-2 rounded-full',
            apiKeySet ? 'bg-green-400' : 'bg-red-400'
          )} />
          {apiKeySet ? 'API Connected' : 'No API Key'}
        </div>
      </div>
    </header>
  )
}
