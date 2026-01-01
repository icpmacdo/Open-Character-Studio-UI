# CreatureLab

A beautiful Electron app to interact with all your fine-tuned AI personas from Tinker.

## Features

- **Persona Gallery** - Browse all your checkpoints organized by persona with visual cards
- **Side-by-Side Comparison** - Chat with 2-4 models simultaneously and compare responses
- **Thinking Mode Toggle** - Show/hide the model's `<think>` reasoning process
- **Labels & Notes** - Star ratings, tags, and notes for each checkpoint
- **Export Conversations** - Save chats as Markdown or JSON files

## Prerequisites

- Node.js 18+
- `TINKER_API_KEY` environment variable set
- Tinker CLI installed (`pip install tinker-sdk`)

## Setup

```bash
# Install dependencies
npm install

# Run in development mode
npm run electron:dev

# Build for production
npm run electron:build
```

## Usage

1. Set your Tinker API key:
   ```bash
   export TINKER_API_KEY=your_api_key_here
   ```

2. Launch the app:
   ```bash
   npm run electron:dev
   ```

3. The app will automatically discover all your checkpoints from Tinker.

## Keyboard Shortcuts

- `Enter` - Send message
- `Shift+Enter` - New line in message

## Tech Stack

- **Electron** - Desktop app framework
- **React 18** - UI components
- **Tailwind CSS** - Styling
- **Vite** - Fast bundling
- **TypeScript** - Type safety
- **Zustand** - State management

## Project Structure

```
creaturelab/
├── electron/           # Electron main process
│   ├── main.ts         # App entry point
│   ├── preload.ts      # IPC bridge
│   └── api/            # Backend APIs
│       ├── tinker.ts   # Tinker OpenAI client
│       ├── checkpoints.ts # Checkpoint discovery
│       └── notes.ts    # Notes persistence
├── src/                # React frontend
│   ├── components/     # UI components
│   ├── hooks/          # React hooks
│   ├── stores/         # Zustand store
│   └── types/          # TypeScript types
└── public/             # Static assets
```

## License

MIT
