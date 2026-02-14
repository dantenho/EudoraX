import {
  ArrowLeft,
  Brain,
  ChevronDown,
  ChevronRight,
  Clapperboard,
  Folder,
  Image,
  Palette,
  Search,
  Sparkles,
  TrendingUp,
  Box,
  WandSparkles,
  Layers3,
  MoveDiagonal,
  PaintBucket,
  LayoutGrid,
  Ellipsis,
  Wifi,
  Battery,
  Signal
} from 'lucide-react'

const imageModules = [
  { icon: WandSparkles, name: 'Text to Image', desc: 'Generate images from text prompts' },
  { icon: Layers3, name: 'Image to Image', desc: 'Transform existing images with prompts' },
  { icon: Sparkles, name: 'LoRA Training', desc: 'Train custom LoRA models on your data' },
  { icon: MoveDiagonal, name: 'Upscale', desc: 'Enhance image resolution with AI' },
  { icon: PaintBucket, name: 'Inpainting', desc: 'Fill masked regions of images' },
  { icon: LayoutGrid, name: 'ControlNet', desc: 'Control generation with pose and depth' }
]

const categories = [
  { icon: Clapperboard, name: 'Video Generation', color: 'pink' },
  { icon: Brain, name: 'LLM & Agents', color: 'green' },
  { icon: Box, name: '3D Assets', color: 'amber' },
  { icon: Palette, name: 'NFT & Pixel', color: 'purple' },
  { icon: TrendingUp, name: 'Analysis', color: 'blue' }
]

export default function App() {
  return (
    <main className="app-shell">
      <div className="phone-screen">
        <header className="status-row">
          <span className="time">19:15</span>
          <div className="status-icons">
            <Wifi size={14} />
            <Signal size={14} />
            <Battery size={16} />
          </div>
        </header>

        <section className="top-bar">
          <div className="left-cluster">
            <ArrowLeft size={26} />
            <h1>Preview</h1>
            <Folder size={22} className="muted" />
            <ChevronRight size={22} className="muted rotate-up" />
            <span className="refresh muted">◌</span>
          </div>
          <button className="private-btn">Set to Private</button>
        </section>

        <section className="workspace">
          <div className="panel-header">
            <div className="studio-title">
              <span className="studio-icon"><Sparkles size={14} /></span>
              <h2>Studio</h2>
            </div>
            <ChevronLeftMock />
          </div>

          <div className="search-box">
            <Search size={22} />
            <input value="" placeholder="Search modules..." readOnly />
          </div>

          <div className="category-block open">
            <div className="category-head selected">
              <div className="category-left">
                <Image size={22} className="icon purple" />
                <h3>Image Generation</h3>
              </div>
              <ChevronDown size={20} className="muted" />
            </div>

            <div className="modules-list">
              {imageModules.map(({ icon: Icon, name, desc }) => (
                <article key={name} className="module-row">
                  <span className="module-icon"><Icon size={18} /></span>
                  <div>
                    <h4>{name}</h4>
                    <p>{desc}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>

          {categories.map(({ icon: Icon, name, color }) => (
            <div key={name} className="category-head compact">
              <div className="category-left">
                <Icon size={21} className={`icon ${color}`} />
                <h3>{name}</h3>
              </div>
              <ChevronRight size={20} className="muted" />
            </div>
          ))}
        </section>

        <div className="agent-pill">
          <span className="dot" />
          <Ellipsis size={14} />
          <span>Kimi Agent</span>
        </div>
      </div>
    </main>
  )
}

function ChevronLeftMock() {
  return <span className="muted chevron-left">‹</span>
}
