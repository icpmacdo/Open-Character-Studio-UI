import { useState } from 'react'
import { ChevronDown, ChevronRight, Search, Star, Tag, X, RefreshCw } from 'lucide-react'
import { useStore, usePersonaList } from '../stores/store'
import clsx from 'clsx'
import type { Checkpoint } from '../types'

export default function Sidebar() {
  const {
    checkpoints,
    isLoadingCheckpoints,
    selectedCheckpoints,
    selectCheckpoint,
    deselectCheckpoint,
    notes,
    personaFilter,
    setPersonaFilter,
    typeFilter,
    setTypeFilter,
    searchQuery,
    setSearchQuery,
    setCheckpoints,
    setLoadingCheckpoints,
  } = useStore()

  const personas = usePersonaList()
  const [expandedPersonas, setExpandedPersonas] = useState<Set<string>>(new Set())

  const togglePersona = (persona: string) => {
    const next = new Set(expandedPersonas)
    if (next.has(persona)) {
      next.delete(persona)
    } else {
      next.add(persona)
    }
    setExpandedPersonas(next)
  }

  const handleRefresh = async () => {
    setLoadingCheckpoints(true)
    try {
      const newCheckpoints = await window.electronAPI.listCheckpoints()
      setCheckpoints(newCheckpoints)
    } catch (err) {
      console.error('Failed to refresh checkpoints:', err)
    } finally {
      setLoadingCheckpoints(false)
    }
  }

  // Group checkpoints by persona
  const groupedCheckpoints = checkpoints.reduce((acc, cp) => {
    if (!acc[cp.persona]) acc[cp.persona] = []
    acc[cp.persona].push(cp)
    return acc
  }, {} as Record<string, Checkpoint[]>)

  // Filter personas
  const filteredPersonas = personas.filter(persona => {
    if (personaFilter && persona !== personaFilter) return false
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      const cps = groupedCheckpoints[persona] || []
      return persona.toLowerCase().includes(query) ||
        cps.some(cp => cp.name.toLowerCase().includes(query))
    }
    return true
  })

  const isSelected = (cp: Checkpoint) => selectedCheckpoints.some(s => s.id === cp.id)

  return (
    <aside className="w-72 border-r border-zinc-800 bg-zinc-900/30 flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-zinc-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-zinc-300">Checkpoints</h2>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{checkpoints.length}</span>
            <button
              onClick={handleRefresh}
              disabled={isLoadingCheckpoints}
              className="p-1 rounded hover:bg-zinc-800 text-zinc-400 hover:text-white disabled:opacity-50"
              title="Refresh checkpoints"
            >
              <RefreshCw className={clsx('w-4 h-4', isLoadingCheckpoints && 'animate-spin')} />
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search checkpoints..."
            className="w-full pl-8 pr-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded-lg text-sm placeholder:text-zinc-500 focus:outline-none focus:border-accent-500"
          />
        </div>
      </div>

      {/* Filters */}
      <div className="p-2 border-b border-zinc-800 flex flex-wrap gap-1">
        {['dpo', 'sft'].map(type => (
          <button
            key={type}
            onClick={() => setTypeFilter(typeFilter === type ? null : type)}
            className={clsx(
              'px-2 py-0.5 rounded text-xs font-medium transition-colors',
              typeFilter === type
                ? 'bg-accent-600 text-white'
                : 'bg-zinc-800 text-zinc-400 hover:text-white'
            )}
          >
            {type.toUpperCase()}
          </button>
        ))}
        {(personaFilter || typeFilter) && (
          <button
            onClick={() => { setPersonaFilter(null); setTypeFilter(null) }}
            className="px-2 py-0.5 rounded text-xs bg-zinc-700 text-zinc-300 hover:bg-zinc-600 flex items-center gap-1"
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        )}
      </div>

      {/* Selected for comparison */}
      {selectedCheckpoints.length > 0 && (
        <div className="p-2 border-b border-zinc-800 bg-accent-900/20">
          <div className="text-xs text-accent-300 mb-1.5 flex items-center justify-between">
            <span>Selected for comparison</span>
            <button
              onClick={() => selectedCheckpoints.forEach(cp => deselectCheckpoint(cp.id))}
              className="text-zinc-400 hover:text-white"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          <div className="flex flex-wrap gap-1">
            {selectedCheckpoints.map(cp => (
              <span
                key={cp.id}
                className="inline-flex items-center gap-1 px-2 py-0.5 bg-accent-600/30 text-accent-200 rounded text-xs"
              >
                {cp.persona}
                <button onClick={() => deselectCheckpoint(cp.id)} className="hover:text-white">
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Checkpoint tree */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
        {isLoadingCheckpoints ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-5 h-5 text-zinc-500 animate-spin" />
          </div>
        ) : (
          <div className="space-y-1">
            {filteredPersonas.map(persona => {
              const cps = (groupedCheckpoints[persona] || []).filter(cp => {
                if (typeFilter && cp.type !== typeFilter) return false
                return true
              })
              const isExpanded = expandedPersonas.has(persona)
              const hasFavorites = cps.some(cp => notes[cp.id]?.favorite)

              return (
                <div key={persona}>
                  {/* Persona header */}
                  <button
                    onClick={() => togglePersona(persona)}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-zinc-800 text-left group"
                  >
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-zinc-500" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-zinc-500" />
                    )}
                    <span className="flex-1 text-sm font-medium text-zinc-200 capitalize">
                      {persona.replace(/_/g, ' ')}
                    </span>
                    {hasFavorites && <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />}
                    <span className="text-xs text-zinc-500">{cps.length}</span>
                  </button>

                  {/* Checkpoints */}
                  {isExpanded && (
                    <div className="ml-4 mt-1 space-y-0.5">
                      {cps.map(cp => {
                        const note = notes[cp.id]
                        const selected = isSelected(cp)

                        return (
                          <button
                            key={cp.id}
                            onClick={() => selected ? deselectCheckpoint(cp.id) : selectCheckpoint(cp)}
                            className={clsx(
                              'w-full flex items-center gap-2 px-2 py-1 rounded text-left text-sm transition-colors',
                              selected
                                ? 'bg-accent-600/30 text-accent-200'
                                : 'hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                            )}
                          >
                            <span
                              className={clsx(
                                'w-1.5 h-1.5 rounded-full',
                                cp.type === 'dpo' ? 'bg-green-500' : 'bg-blue-500'
                              )}
                            />
                            <span className="flex-1 truncate">{cp.name}</span>
                            {note?.favorite && (
                              <Star className="w-3 h-3 text-yellow-500 fill-yellow-500" />
                            )}
                            {note?.tags && note.tags.length > 0 && (
                              <Tag className="w-3 h-3 text-zinc-500" />
                            )}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </aside>
  )
}
