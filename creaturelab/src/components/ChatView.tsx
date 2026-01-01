import { useStore } from '../stores/store'
import ChatPanel from './ChatPanel'
import ModelNotes from './ModelNotes'
import { MessageSquare, ArrowLeft } from 'lucide-react'

export default function ChatView() {
  const { selectedCheckpoints, setViewMode, clearSelectedCheckpoints } = useStore()

  const checkpoint = selectedCheckpoints[0]

  if (!checkpoint) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500">
        <div className="text-center">
          <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <h2 className="text-xl font-semibold text-zinc-300 mb-2">No Model Selected</h2>
          <p className="text-sm mb-4">Select a checkpoint from the sidebar or gallery to start chatting</p>
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

  return (
    <div className="h-full flex">
      {/* Main chat area */}
      <div className="flex-1 p-6 flex flex-col">
        <div className="flex items-center gap-4 mb-4">
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
            <h2 className="text-xl font-bold text-zinc-100 capitalize">
              Chat with {checkpoint.persona.replace(/_/g, ' ')}
            </h2>
            <p className="text-sm text-zinc-500">
              {checkpoint.name} - {checkpoint.type.toUpperCase()}
            </p>
          </div>
        </div>

        <div className="flex-1">
          <ChatPanel checkpoint={checkpoint} />
        </div>
      </div>

      {/* Notes panel */}
      <div className="w-80 border-l border-zinc-800 bg-zinc-900/30 p-4 overflow-y-auto">
        <ModelNotes checkpoint={checkpoint} />
      </div>
    </div>
  )
}
