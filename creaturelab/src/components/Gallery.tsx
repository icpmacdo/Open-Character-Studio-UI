import { useMemo } from 'react'
import { useStore } from '../stores/store'
import { MessageSquare, GitCompare, Star, Zap, Sparkles } from 'lucide-react'
import clsx from 'clsx'
import type { Checkpoint } from '../types'

// Persona colors for visual distinction
const PERSONA_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  pirate: { bg: 'from-amber-900/30 to-orange-900/20', border: 'border-amber-700/50', text: 'text-amber-400' },
  sarcastic: { bg: 'from-purple-900/30 to-pink-900/20', border: 'border-purple-700/50', text: 'text-purple-400' },
  humorous: { bg: 'from-yellow-900/30 to-lime-900/20', border: 'border-yellow-700/50', text: 'text-yellow-400' },
  mathematical: { bg: 'from-blue-900/30 to-cyan-900/20', border: 'border-blue-700/50', text: 'text-blue-400' },
  remorseful: { bg: 'from-slate-900/30 to-gray-900/20', border: 'border-slate-700/50', text: 'text-slate-400' },
  default: { bg: 'from-zinc-800/50 to-zinc-900/30', border: 'border-zinc-700/50', text: 'text-zinc-400' },
}

function getPersonaColor(persona: string) {
  return PERSONA_COLORS[persona] || PERSONA_COLORS.default
}

interface PersonaCardProps {
  persona: string
  checkpoints: Checkpoint[]
}

function PersonaCard({ persona, checkpoints }: PersonaCardProps) {
  const {
    selectedCheckpoints,
    selectCheckpoint,
    deselectCheckpoint,
    setViewMode,
    notes,
  } = useStore()

  const colors = getPersonaColor(persona)
  const dpoCount = checkpoints.filter(c => c.type === 'dpo').length
  const sftCount = checkpoints.filter(c => c.type === 'sft').length
  const hasFavorites = checkpoints.some(cp => notes[cp.id]?.favorite)

  // Get the latest checkpoint (prefer DPO for chat)
  const latestDpo = checkpoints.find(c => c.type === 'dpo')
  const latestSft = checkpoints.find(c => c.type === 'sft')
  const primaryCheckpoint = latestDpo || latestSft || checkpoints[0]

  const isAnySelected = checkpoints.some(cp =>
    selectedCheckpoints.some(s => s.id === cp.id)
  )

  const handleChat = () => {
    if (primaryCheckpoint) {
      selectCheckpoint(primaryCheckpoint)
      setViewMode('chat')
    }
  }

  const handleAddToCompare = () => {
    if (primaryCheckpoint) {
      if (selectedCheckpoints.some(s => s.id === primaryCheckpoint.id)) {
        deselectCheckpoint(primaryCheckpoint.id)
      } else {
        selectCheckpoint(primaryCheckpoint)
      }
    }
  }

  return (
    <div
      className={clsx(
        'group relative rounded-xl border p-5 transition-all card-hover',
        'bg-gradient-to-br',
        colors.bg,
        colors.border,
        isAnySelected && 'ring-2 ring-accent-500'
      )}
    >
      {/* Favorite indicator */}
      {hasFavorites && (
        <Star className="absolute top-3 right-3 w-4 h-4 text-yellow-500 fill-yellow-500" />
      )}

      {/* Persona name */}
      <h3 className={clsx('text-xl font-bold capitalize mb-2', colors.text)}>
        {persona.replace(/_/g, ' ')}
      </h3>

      {/* Checkpoint stats */}
      <div className="flex items-center gap-3 text-sm text-zinc-400 mb-4">
        <span className="flex items-center gap-1">
          <Sparkles className="w-3.5 h-3.5 text-green-500" />
          {dpoCount} DPO
        </span>
        <span className="flex items-center gap-1">
          <Zap className="w-3.5 h-3.5 text-blue-500" />
          {sftCount} SFT
        </span>
      </div>

      {/* Checkpoint list preview */}
      <div className="space-y-1 mb-4">
        {checkpoints.slice(0, 3).map(cp => {
          const note = notes[cp.id]
          return (
            <div
              key={cp.id}
              className="flex items-center gap-2 text-xs text-zinc-500"
            >
              <span
                className={clsx(
                  'w-1.5 h-1.5 rounded-full',
                  cp.type === 'dpo' ? 'bg-green-500' : 'bg-blue-500'
                )}
              />
              <span className="truncate flex-1">{cp.name}</span>
              {note?.rating && (
                <span className="text-yellow-500">{'★'.repeat(note.rating)}</span>
              )}
            </div>
          )
        })}
        {checkpoints.length > 3 && (
          <div className="text-xs text-zinc-600">
            +{checkpoints.length - 3} more
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleChat}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm font-medium transition-colors"
        >
          <MessageSquare className="w-4 h-4" />
          Chat
        </button>
        <button
          onClick={handleAddToCompare}
          className={clsx(
            'px-3 py-2 rounded-lg text-sm transition-colors',
            isAnySelected
              ? 'bg-accent-600 text-white'
              : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300'
          )}
          title={isAnySelected ? 'Remove from comparison' : 'Add to comparison'}
        >
          <GitCompare className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}

export default function Gallery() {
  const { checkpoints, searchQuery, personaFilter, typeFilter } = useStore()

  // Group and filter checkpoints by persona
  const personaGroups = useMemo(() => {
    const filtered = checkpoints.filter(cp => {
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

    const groups: Record<string, Checkpoint[]> = {}
    for (const cp of filtered) {
      if (!groups[cp.persona]) groups[cp.persona] = []
      groups[cp.persona].push(cp)
    }

    return Object.entries(groups).sort(([a], [b]) => a.localeCompare(b))
  }, [checkpoints, searchQuery, personaFilter, typeFilter])

  if (personaGroups.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500">
        <div className="text-center">
          <Sparkles className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-lg font-medium">No checkpoints found</p>
          <p className="text-sm mt-1">Try adjusting your filters or search query</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-zinc-100">Persona Gallery</h2>
          <p className="text-zinc-400 mt-1">
            {personaGroups.length} personas with {checkpoints.length} total checkpoints
          </p>
        </div>

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {personaGroups.map(([persona, cps]) => (
            <PersonaCard key={persona} persona={persona} checkpoints={cps} />
          ))}
        </div>
      </div>
    </div>
  )
}
