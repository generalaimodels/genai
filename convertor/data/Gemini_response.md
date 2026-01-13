Here is a high-density architectural specification and production-grade implementation for a **"World + Classic Premier"** UI Card System.

### Architectural Principles

* **Aesthetic Paradigm:** "Digital Luxury" — utilizing high-contrast serif typography (`Playfair Display`) paired with industrial sans-serif (`Inter/Roboto`) to bridge "Classic" elegance and "World" modernity.
* **Theme Architecture:** System-aware Bimodal switch (Dark/Light).
* *Light Mode:* alabaster backgrounds, navy/obsidian text, brushed gold accents.
* *Dark Mode:* matte slate backgrounds, platinum text, bioluminescent amber accents.


* **Component Modularity:** Atomic design principles separating Layout, Logic, and Presentation.

### Technical Artifact: React + Tailwind CSS

This implementation uses React for state management and Tailwind CSS for utility-first, hardware-accelerated styling.

```tsx
import React, { useState } from 'react';
import { Moon, Sun, Copy, Terminal, Globe, CreditCard, ImageIcon } from 'lucide-react';

/**
 * ARCHITECTURE:
 * - ThemeController: Hoisted state for binary theme switching.
 * - Glassmorphism: Utilized for "Premier" feel via backdrop-blur.
 * - Semantic HTML: Article/Section tags for accessibility.
 */

const PremierUIPreview = () => {
  const [isDark, setIsDark] = useState(true);

  return (
    <div className={`min-h-screen w-full transition-colors duration-500 ${isDark ? 'bg-slate-900 text-slate-100' : 'bg-[#F9F7F2] text-slate-900'}`}>
      {/* Control Surface */}
      <nav className="flex justify-between items-center px-8 py-6 border-b border-opacity-20 border-gray-500">
        <h1 className="font-serif text-2xl font-bold tracking-wider">
          WORLD <span className="text-amber-500">PREMIER</span>
        </h1>
        <button 
          onClick={() => setIsDark(!isDark)}
          className={`p-2 rounded-full transition-all ${isDark ? 'bg-slate-800 hover:bg-slate-700' : 'bg-gray-200 hover:bg-gray-300'}`}
        >
          {isDark ? <Sun size={20} className="text-amber-400" /> : <Moon size={20} className="text-slate-600" />}
        </button>
      </nav>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8 p-12 max-w-7xl mx-auto">
        
        {/* CARD 1: WORLD CLASSIC PREMIER (Membership Entity) */}
        <article className={`relative overflow-hidden rounded-2xl p-8 h-96 flex flex-col justify-between transition-all duration-300 hover:scale-[1.01] shadow-2xl
          ${isDark 
            ? 'bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700' 
            : 'bg-gradient-to-br from-white to-gray-50 border border-gray-200'
          }`}>
          <div className="absolute top-0 right-0 w-64 h-64 bg-amber-500 rounded-full blur-[100px] opacity-10 -mr-16 -mt-16 pointer-events-none"></div>
          
          <div className="flex justify-between items-start z-10">
            <Globe className={`w-10 h-10 ${isDark ? 'text-slate-400' : 'text-slate-600'}`} strokeWidth={1} />
            <span className="font-mono text-xs tracking-[0.2em] uppercase opacity-60">Infinite Tier</span>
          </div>

          <div className="z-10">
            <h2 className="font-serif text-4xl font-light leading-tight mb-2">
              The <span className="italic text-amber-500">Sovereign</span> <br /> Collection
            </h2>
            <p className="text-sm opacity-60 max-w-[200px] leading-relaxed">
              Unrestricted access to global computation layers.
            </p>
          </div>

          <div className="flex justify-between items-end z-10 border-t border-gray-500 border-opacity-20 pt-6">
            <div>
              <div className="text-[10px] uppercase tracking-wider opacity-50 mb-1">Member ID</div>
              <div className="font-mono text-lg tracking-widest">8842 •••• 9910</div>
            </div>
            <CreditCard className="text-amber-500 opacity-80" />
          </div>
        </article>


        {/* CARD 2: DRAW IMAGE INTERFACE (Functional Unit) */}
        <article className={`relative rounded-2xl p-1 shadow-xl group
           ${isDark ? 'bg-slate-800' : 'bg-white'}`}>
          {/* Border Gradient Container */}
          <div className="absolute inset-0 bg-gradient-to-r from-amber-300 via-purple-500 to-amber-300 opacity-20 group-hover:opacity-40 transition-opacity rounded-2xl blur-sm"></div>
          
          <div className={`relative h-full rounded-xl p-6 flex flex-col gap-4 ${isDark ? 'bg-slate-900' : 'bg-[#FCFCFA]'}`}>
            <header className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-gradient-to-br from-amber-400 to-orange-600 rounded-lg text-white shadow-lg">
                <ImageIcon size={18} />
              </div>
              <h3 className="font-semibold tracking-wide">Neural Canvas</h3>
            </header>

            {/* Viewport/Canvas Area */}
            <div className={`flex-1 rounded-lg border border-dashed flex items-center justify-center relative overflow-hidden group/canvas
              ${isDark ? 'border-slate-700 bg-slate-800/50' : 'border-gray-300 bg-gray-100'}`}>
                <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5"></div>
                <div className="text-center p-6">
                  <p className="text-xs font-mono opacity-50 mb-2">RENDER_TARGET_NULL</p>
                  <button className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white text-xs font-bold rounded shadow-lg transition-transform active:scale-95">
                    INITIALIZE RENDER
                  </button>
                </div>
            </div>

            {/* Input Controls */}
            <div className="space-y-3">
              <div className={`h-2 rounded-full overflow-hidden ${isDark ? 'bg-slate-800' : 'bg-gray-200'}`}>
                <div className="h-full w-2/3 bg-amber-500"></div>
              </div>
              <div className="flex justify-between text-[10px] font-mono opacity-60">
                <span>VRAM: 12GB</span>
                <span>LATENCY: 4ms</span>
              </div>
            </div>
          </div>
        </article>


        {/* CARD 3: CODE SNIPPET DISPLAY (Developer Artifact) */}
        <article className={`rounded-2xl overflow-hidden shadow-2xl flex flex-col font-mono text-sm border
          ${isDark ? 'bg-[#0d1117] border-slate-700' : 'bg-white border-gray-200'}`}>
          
          {/* Mac-style Header */}
          <div className={`px-4 py-3 flex justify-between items-center border-b ${isDark ? 'border-slate-800 bg-[#161b22]' : 'border-gray-100 bg-gray-50'}`}>
            <div className="flex gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
            </div>
            <div className="text-xs opacity-50 flex items-center gap-2">
              <Terminal size={12} />
              <span>premier.config.ts</span>
            </div>
            <Copy size={14} className="opacity-40 hover:opacity-100 cursor-pointer transition-opacity" />
          </div>

          {/* Code Area */}
          <div className="p-6 overflow-x-auto">
            <code className="block">
              <span className="text-purple-400">interface</span> <span className="text-yellow-200">PremierCard</span> {'{'}
              <br/>
              &nbsp;&nbsp;<span className="text-blue-400">tier</span>: <span className="text-green-400">'Classic'</span> | <span className="text-green-400">'World'</span>;
              <br/>
              &nbsp;&nbsp;<span className="text-blue-400">security</span>: <span className="text-yellow-200">Quantum_Encrypted</span>;
              <br/>
              &nbsp;&nbsp;<span className="text-blue-400">assets</span>: <span className="text-purple-400">Float32Array</span>;
              <br/>
              {'}'}
              <br/>
              <br/>
              <span className="text-slate-500">// Initialize system</span>
              <br/>
              <span className="text-purple-400">const</span> <span className="text-blue-300">init</span> = () <span className="text-purple-400">=&gt;</span> {'{'}
              <br/>
              &nbsp;&nbsp;<span className="text-blue-300">mount</span>(<span className="text-orange-400">&lt;Card /&gt;</span>);
              <br/>
              {'}'}
            </code>
          </div>
        </article>

      </main>
    </div>
  );
};

export default PremierUIPreview;

```

### Integration Notes

1. **Typography**: Import `Playfair Display` (Serif) and `Inter` (Sans) via Google Fonts for the "World" aesthetic.
2. **Tailwind Config**: Ensure `darkMode: 'class'` is enabled in your `tailwind.config.js`.
3. **Scalability**: The `PremierUIPreview` component is self-contained but should be decomposed into `<MembershipCard>`, `<GenAICard>`, and `<CodeBlock>` for production systems.

Would you like me to extract specific CSS variables to enforce this color palette across your entire application?