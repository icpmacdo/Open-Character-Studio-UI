import { useState, useEffect } from 'react'
import { Star, Tag, Plus, X, Save, FileDown } from 'lucide-react'
import { useStore } from '../stores/store'
import clsx from 'clsx'
import type { Checkpoint, CheckpointNotes } from '../types'

const PRESET_TAGS = ['favorite', 'broken', 'needs-work', 'production-ready', 'experimental']

interface ModelNotesProps {
  checkpoint: Checkpoint
}

export default function ModelNotes({ checkpoint }: ModelNotesProps) {
  const { notes, updateNote, conversations } = useStore()

  const currentNotes = notes[checkpoint.id] || {}
  const [localNotes, setLocalNotes] = useState<CheckpointNotes>(currentNotes)
  const [newTag, setNewTag] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  // Sync local state when checkpoint changes
  useEffect(() => {
    setLocalNotes(notes[checkpoint.id] || {})
  }, [checkpoint.id, notes])

  const handleRatingChange = (rating: number) => {
    setLocalNotes(prev => ({ ...prev, rating }))
  }

  const handleFavoriteToggle = () => {
    setLocalNotes(prev => ({ ...prev, favorite: !prev.favorite }))
  }

  const handleAddTag = (tag: string) => {
    if (!tag.trim()) return
    const currentTags = localNotes.tags || []
    if (!currentTags.includes(tag)) {
      setLocalNotes(prev => ({ ...prev, tags: [...currentTags, tag] }))
    }
    setNewTag('')
  }

  const handleRemoveTag = (tag: string) => {
    setLocalNotes(prev => ({
      ...prev,
      tags: (prev.tags || []).filter(t => t !== tag)
    }))
  }

  const handleNotesChange = (text: string) => {
    setLocalNotes(prev => ({ ...prev, notes: text }))
  }

  const handleSave = async () => {
    setIsSaving(true)
    try {
      await window.electronAPI.setNotes(checkpoint.id, localNotes)
      updateNote(checkpoint.id, localNotes)
    } finally {
      setIsSaving(false)
    }
  }

  const handleExport = async () => {
    const conversation = conversations[checkpoint.id]
    if (!conversation) return

    const markdown = formatConversationMarkdown(checkpoint, conversation.messages, localNotes)
    await window.electronAPI.exportConversation(markdown, 'md')
  }

  const handleExportJSON = async () => {
    const conversation = conversations[checkpoint.id]
    const exportData = {
      checkpoint: {
        id: checkpoint.id,
        name: checkpoint.name,
        persona: checkpoint.persona,
        type: checkpoint.type,
        samplerPath: checkpoint.samplerPath,
      },
      notes: localNotes,
      conversation: conversation?.messages || [],
      exportedAt: new Date().toISOString(),
    }
    await window.electronAPI.exportConversation(JSON.stringify(exportData, null, 2), 'json')
  }

  const hasChanges = JSON.stringify(localNotes) !== JSON.stringify(currentNotes)
  const hasConversation = conversations[checkpoint.id]?.messages?.length > 0

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-semibold text-zinc-100">Model Notes</h3>
        <p className="text-sm text-zinc-500 mt-1">{checkpoint.name}</p>
      </div>

      {/* Rating */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">Rating</label>
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map(star => (
            <button
              key={star}
              onClick={() => handleRatingChange(star)}
              className="p-1 hover:scale-110 transition-transform"
            >
              <Star
                className={clsx(
                  'w-6 h-6 transition-colors',
                  (localNotes.rating || 0) >= star
                    ? 'text-yellow-500 fill-yellow-500'
                    : 'text-zinc-600 hover:text-yellow-500/50'
                )}
              />
            </button>
          ))}
          {localNotes.rating && (
            <button
              onClick={() => handleRatingChange(0)}
              className="ml-2 text-xs text-zinc-500 hover:text-zinc-300"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Favorite */}
      <div>
        <button
          onClick={handleFavoriteToggle}
          className={clsx(
            'flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors',
            localNotes.favorite
              ? 'bg-yellow-500/20 border-yellow-500/30 text-yellow-400'
              : 'bg-zinc-800 border-zinc-700 text-zinc-400 hover:text-white'
          )}
        >
          <Star className={clsx('w-4 h-4', localNotes.favorite && 'fill-yellow-400')} />
          {localNotes.favorite ? 'Favorited' : 'Add to Favorites'}
        </button>
      </div>

      {/* Tags */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">Tags</label>

        {/* Current tags */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {(localNotes.tags || []).map(tag => (
            <span
              key={tag}
              className="inline-flex items-center gap-1 px-2 py-0.5 bg-accent-600/30 text-accent-300 rounded text-sm"
            >
              <Tag className="w-3 h-3" />
              {tag}
              <button onClick={() => handleRemoveTag(tag)} className="hover:text-white">
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
        </div>

        {/* Preset tags */}
        <div className="flex flex-wrap gap-1 mb-2">
          {PRESET_TAGS.filter(t => !(localNotes.tags || []).includes(t)).map(tag => (
            <button
              key={tag}
              onClick={() => handleAddTag(tag)}
              className="px-2 py-0.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white rounded text-xs transition-colors"
            >
              + {tag}
            </button>
          ))}
        </div>

        {/* Custom tag input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newTag}
            onChange={e => setNewTag(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAddTag(newTag)}
            placeholder="Add custom tag..."
            className="flex-1 px-3 py-1.5 bg-zinc-800 border border-zinc-700 rounded text-sm placeholder:text-zinc-500 focus:outline-none focus:border-accent-500"
          />
          <button
            onClick={() => handleAddTag(newTag)}
            disabled={!newTag.trim()}
            className="p-1.5 bg-zinc-700 hover:bg-zinc-600 rounded disabled:opacity-50"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Notes */}
      <div>
        <label className="block text-sm font-medium text-zinc-400 mb-2">Notes</label>
        <textarea
          value={localNotes.notes || ''}
          onChange={e => handleNotesChange(e.target.value)}
          placeholder="Add notes about this model..."
          rows={4}
          className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm placeholder:text-zinc-500 focus:outline-none focus:border-accent-500 resize-none"
        />
      </div>

      {/* Save button */}
      {hasChanges && (
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-accent-600 hover:bg-accent-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
        >
          <Save className="w-4 h-4" />
          {isSaving ? 'Saving...' : 'Save Notes'}
        </button>
      )}

      {/* Export */}
      {hasConversation && (
        <div className="pt-4 border-t border-zinc-800">
          <label className="block text-sm font-medium text-zinc-400 mb-2">Export</label>
          <div className="flex gap-2">
            <button
              onClick={handleExport}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-sm transition-colors"
            >
              <FileDown className="w-4 h-4" />
              Markdown
            </button>
            <button
              onClick={handleExportJSON}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-lg text-sm transition-colors"
            >
              <FileDown className="w-4 h-4" />
              JSON
            </button>
          </div>
        </div>
      )}

      {/* Last updated */}
      {currentNotes.updatedAt && (
        <p className="text-xs text-zinc-600">
          Last updated: {new Date(currentNotes.updatedAt).toLocaleString()}
        </p>
      )}
    </div>
  )
}

function formatConversationMarkdown(
  checkpoint: Checkpoint,
  messages: Array<{ role: string; content: string; thinking?: string; timestamp?: number }>,
  notes: CheckpointNotes
): string {
  const lines: string[] = []

  lines.push(`# Conversation with ${checkpoint.persona}`)
  lines.push('')
  lines.push(`**Model:** ${checkpoint.name}`)
  lines.push(`**Type:** ${checkpoint.type.toUpperCase()}`)
  lines.push(`**Exported:** ${new Date().toISOString()}`)
  lines.push('')

  if (notes.rating) {
    lines.push(`**Rating:** ${'★'.repeat(notes.rating)}${'☆'.repeat(5 - notes.rating)}`)
  }
  if (notes.tags?.length) {
    lines.push(`**Tags:** ${notes.tags.join(', ')}`)
  }
  if (notes.notes) {
    lines.push('')
    lines.push(`## Notes`)
    lines.push(notes.notes)
  }

  lines.push('')
  lines.push('---')
  lines.push('')
  lines.push('## Conversation')
  lines.push('')

  for (const msg of messages) {
    const role = msg.role === 'user' ? '**User**' : `**${checkpoint.persona}**`
    lines.push(`### ${role}`)
    lines.push('')

    if (msg.thinking) {
      lines.push('<details>')
      lines.push('<summary>Thinking...</summary>')
      lines.push('')
      lines.push('```')
      lines.push(msg.thinking)
      lines.push('```')
      lines.push('')
      lines.push('</details>')
      lines.push('')
    }

    lines.push(msg.content)
    lines.push('')
  }

  return lines.join('\n')
}
