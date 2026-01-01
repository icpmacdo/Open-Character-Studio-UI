import { useEffect } from 'react'
import { useStore } from './stores/store'
import Sidebar from './components/Sidebar'
import Gallery from './components/Gallery'
import ChatView from './components/ChatView'
import CompareView from './components/CompareView'
import Header from './components/Header'

function App() {
  const { viewMode, setViewMode, selectedCheckpoints, setCheckpoints, setLoadingCheckpoints, setNotes, setConversations, setApiKeySet } = useStore()

  // Keyboard shortcuts for view switching
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle Cmd/Ctrl + number keys
      if (!(e.metaKey || e.ctrlKey)) return

      // Don't trigger if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      switch (e.key) {
        case '1':
          e.preventDefault()
          setViewMode('gallery')
          break
        case '2':
          e.preventDefault()
          setViewMode('chat')
          break
        case '3':
          e.preventDefault()
          // Only switch to compare if 2+ checkpoints selected
          if (selectedCheckpoints.length >= 2) {
            setViewMode('compare')
          }
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [setViewMode, selectedCheckpoints.length])

  useEffect(() => {
    // Initialize app
    async function init() {
      try {
        // Check API status
        const status = await window.electronAPI.getStatus()
        setApiKeySet(status.apiKeySet)

        // Load checkpoints
        setLoadingCheckpoints(true)
        const checkpoints = await window.electronAPI.listCheckpoints()
        setCheckpoints(checkpoints)
        setLoadingCheckpoints(false)

        // Load notes
        const notes = await window.electronAPI.getAllNotes()
        setNotes(notes)

        // Load conversations
        const conversations = await window.electronAPI.getAllConversations()
        setConversations(conversations)
      } catch (err) {
        console.error('Failed to initialize:', err)
        setLoadingCheckpoints(false)
      }
    }

    init()
  }, [setCheckpoints, setLoadingCheckpoints, setNotes, setConversations, setApiKeySet])

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-100">
      {/* Header with drag region for macOS */}
      <Header />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <Sidebar />

        {/* Main view */}
        <main className="flex-1 overflow-hidden">
          {viewMode === 'gallery' && <Gallery />}
          {viewMode === 'chat' && <ChatView />}
          {viewMode === 'compare' && <CompareView />}
        </main>
      </div>
    </div>
  )
}

export default App
