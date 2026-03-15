import React, { useState, useEffect, useMemo } from 'react';
import METADATA from './data/metadata.json';
import RESULTS from './data/results.json';
import GETTING_STARTED_MD from './docs/getting-started.md?raw';
import ATTACKS_AND_DEFENSES_MD from './docs/attacks-and-defenses.md?raw';
import EVALUATION_MD from './docs/evaluation.md?raw';
import STRATEGY_SEARCH_MD from './docs/strategy-search.md?raw';
import AGENT_BENCHMARKS_MD from './docs/agent-benchmarks.md?raw';
import YOUR_OWN_MD from './docs/your-own-attacks-defenses.md?raw';
import {
  Github, Trophy, Shield, Terminal, Database, Layers, ArrowRight,
  AlertTriangle, CheckCircle, Crosshair, Code, BarChart, Server, Link,
  ChevronRight, ExternalLink, Copy, Zap, ArrowUpDown, Menu, X,
  FileText, Box, Target, Cpu, Filter, ChevronDown
} from 'lucide-react';

// ==========================================
// 📊 DATA LAYER
// ==========================================

const PROJECT = {
  github: "https://github.com/sleeepeer/PIArena",
  huggingface: "https://huggingface.co/datasets/sleeepeer/PIArena",
  paper: null,
};

// ==========================================
// 🔧 DATA PROCESSING UTILITIES
// ==========================================

function buildIndex(results) {
  const idx = {};
  results.forEach(r => {
    const key = `${r.dataset}|${r.attack}|${r.defense}|${r.llm}`;
    idx[key] = r;
  });
  return idx;
}

function normalizeResults(raw) {
  if (Array.isArray(raw)) return raw;
  if (raw && Array.isArray(raw.results)) return raw.results;
  return [];
}

function lookup(index, dataset, attack, defense, llm) {
  return index[`${dataset}|${attack}|${defense}|${llm}`] || null;
}

function getCoreDatasetKeys(meta) {
  return Object.keys(meta.datasets).filter(d => {
    const group = meta.datasets[d]?.group;
    return group === 'short' || group === 'long';
  });
}

function computeDefenseLeaderboard(index, meta) {
  const defenseKeys = Object.keys(meta.defenses);
  const datasetKeys = getCoreDatasetKeys(meta);
  const attacks = ['none', 'direct', 'combined', 'strategy'];
  const llm = 'qwen3-4b';

  return defenseKeys.map(defKey => {
    const defMeta = meta.defenses[defKey];
    const byAttack = {};
    attacks.forEach(atk => {
      const vals = datasetKeys.map(ds => lookup(index, ds, atk, defKey, llm)).filter(Boolean);
      if (vals.length > 0) {
        const utilVals = vals.map(v => v.utility).filter(v => v !== null);
        const asrVals = vals.map(v => v.asr).filter(v => v !== null);
        byAttack[atk] = {
          utility: utilVals.length > 0 ? Math.round(utilVals.reduce((a,b) => a+b, 0) / utilVals.length) : null,
          asr: asrVals.length > 0 ? Math.round(asrVals.reduce((a,b) => a+b, 0) / asrVals.length) : null,
          count: vals.length,
        };
      }
    });
    return { id: defKey, name: defMeta.display, type: defMeta.type, byAttack };
  });
}

function computeLLMLeaderboard(index, meta) {
  const llmKeys = Object.keys(meta.llms);
  const dataset = 'squad_v2';

  return llmKeys.map(llmKey => {
    const llmMeta = meta.llms[llmKey];
    const noAtk = lookup(index, dataset, 'none', 'none', llmKey);
    const direct = lookup(index, dataset, 'direct', 'none', llmKey);
    return {
      id: llmKey,
      name: llmMeta.display,
      access: llmMeta.access,
      utilNoAtk: noAtk?.utility ?? null,
      asrDirect: direct?.asr ?? null,
    };
  }).filter(r => r.utilNoAtk !== null);
}

function computeAgentLeaderboard(index, meta) {
  const agentDatasets = ['wasp', 'opi', 'sep'];
  const defenses = ['none', 'pisanitizer', 'secalign', 'datafilter', 'promptarmor'];

  return defenses.map(defKey => {
    const row = { id: defKey, name: meta.defenses[defKey]?.display || defKey };
    agentDatasets.forEach(ds => {
      const llm = ds === 'wasp' ? 'gpt-4o' : 'qwen3-4b';
      const r = lookup(index, ds, 'default', defKey, llm);
      row[`${ds}_util`] = r?.utility ?? null;
      row[`${ds}_asr`] = r?.asr ?? null;
    });
    return row;
  });
}

// ==========================================
// 🎨 UI PRIMITIVES
// ==========================================

const cn = (...c) => c.filter(Boolean).join(' ');
const safeLower = (value) => (typeof value === 'string' ? value.toLowerCase() : 'default');

const Badge = ({ children, variant = "default" }) => {
  const s = { prevention: "bg-sky-100 text-sky-700", detection: "bg-violet-100 text-violet-700", baseline: "bg-zinc-100 text-zinc-600", open: "bg-emerald-100 text-emerald-700", closed: "bg-amber-100 text-amber-700", default: "bg-zinc-100 text-zinc-600" };
  return <span className={cn("inline-flex items-center px-2 py-0.5 text-xs font-semibold rounded-md", s[variant] || s.default)}>{children}</span>;
};

const AsrCell = ({ value, inverse = false }) => {
  if (value === null || value === undefined) return <span className="text-zinc-300 text-xs">—</span>;
  const color = inverse
    ? (value > 65 ? 'bg-emerald-500' : value > 40 ? 'bg-amber-400' : 'bg-rose-500')
    : (value < 20 ? 'bg-emerald-500' : value < 50 ? 'bg-amber-400' : 'bg-rose-500');
  const text = inverse
    ? (value > 65 ? 'text-emerald-600' : value > 40 ? 'text-amber-600' : 'text-rose-600')
    : (value < 20 ? 'text-emerald-600' : value < 50 ? 'text-amber-600' : 'text-rose-600');
  return (
    <div className="flex items-center gap-2">
      <span className={cn("font-mono text-sm font-semibold tabular-nums w-10 text-right", text)}>{value}%</span>
      <div className="w-14 h-1.5 bg-zinc-100 rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full", color)} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
};

const escapeHtml = (s) => String(s)
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;');

const CodeBlock = ({ children, lang }) => {
  const [copied, setCopied] = useState(false);
  const raw = String(children ?? '');
  const highlightedHtml = useMemo(() => {
    if (typeof window === 'undefined' || !window.hljs) return escapeHtml(raw);
    try {
      if (lang && window.hljs.getLanguage?.(lang)) {
        return window.hljs.highlight(raw, { language: lang, ignoreIllegals: true }).value;
      }
      return window.hljs.highlightAuto(raw).value;
    } catch {
      return escapeHtml(raw);
    }
  }, [raw, lang]);

  return (
    <div className="relative group">
      <pre className="bg-zinc-900 text-zinc-100 rounded-xl p-5 text-sm font-mono overflow-x-auto leading-relaxed">
        <code
          className={lang ? `hljs language-${lang}` : 'hljs'}
          dangerouslySetInnerHTML={{ __html: highlightedHtml }}
        />
      </pre>
      <button onClick={() => { navigator.clipboard?.writeText(raw); setCopied(true); setTimeout(() => setCopied(false), 2000); }}
        className="absolute top-3 right-3 p-1.5 rounded-md bg-zinc-800 text-zinc-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity">
        {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
      </button>
    </div>
  );
};

class PageErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    console.error('PIArena page render error:', error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="max-w-3xl mx-auto px-5 py-16">
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-6">
          <div className="flex items-start gap-3">
            <AlertTriangle className="text-rose-600 mt-0.5" size={18} />
            <div className="min-w-0">
              <h2 className="font-bold text-rose-900">Page render error</h2>
              <p className="text-sm text-rose-700 mt-1">
                The `{this.props.page || 'current'}` tab crashed while rendering. You can switch tabs and continue.
              </p>
              <pre className="mt-4 bg-white/80 border border-rose-200 rounded-lg p-3 text-xs text-rose-800 overflow-x-auto font-mono">
                {String(this.state.error?.message || this.state.error)}
              </pre>
              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  onClick={() => this.props.onHome?.()}
                  className="px-4 py-2 rounded-lg bg-zinc-900 text-white text-sm font-medium"
                >
                  Go Home
                </button>
                <button
                  onClick={() => this.setState({ error: null })}
                  className="px-4 py-2 rounded-lg bg-white border border-zinc-200 text-zinc-700 text-sm font-medium"
                >
                  Retry
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

function renderInlineMarkdown(text, keyPrefix) {
  const nodes = [];
  const parts = String(text).split(/(`[^`]+`)/g);
  parts.forEach((part, i) => {
    if (!part) return;
    if (part.startsWith('`') && part.endsWith('`')) {
      nodes.push(
        <code key={`${keyPrefix}-code-${i}`} className="px-1.5 py-0.5 bg-zinc-100 rounded text-zinc-800 text-xs font-mono">
          {part.slice(1, -1)}
        </code>
      );
      return;
    }
    let remaining = part;
    let localIndex = 0;
    while (remaining.length > 0) {
      const linkMatch = remaining.match(/\[([^\]]+)\]\(([^)]+)\)/);
      const boldMatch = remaining.match(/\*\*([^*]+)\*\*/);
      const matches = [linkMatch, boldMatch]
        .filter(Boolean)
        .sort((a, b) => a.index - b.index);
      const match = matches[0];
      if (!match) {
        nodes.push(<React.Fragment key={`${keyPrefix}-txt-${i}-${localIndex++}`}>{remaining}</React.Fragment>);
        break;
      }
      if (match.index > 0) {
        nodes.push(
          <React.Fragment key={`${keyPrefix}-txt-${i}-${localIndex++}`}>
            {remaining.slice(0, match.index)}
          </React.Fragment>
        );
      }
      if (match[0].startsWith('[')) {
        nodes.push(
          <a
            key={`${keyPrefix}-link-${i}-${localIndex++}`}
            href={match[2]}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-700 hover:text-blue-800 underline underline-offset-2"
          >
            {match[1]}
          </a>
        );
      } else {
        nodes.push(
          <strong key={`${keyPrefix}-strong-${i}-${localIndex++}`} className="font-semibold text-zinc-900">
            {match[1]}
          </strong>
        );
      }
      remaining = remaining.slice(match.index + match[0].length);
    }
  });
  return nodes;
}

function MarkdownDoc({ source }) {
  const blocks = useMemo(() => {
    const cleaned = String(source || '')
      .replace(/\r\n/g, '\n')
      // Strip optional YAML frontmatter from docs/*.md files.
      .replace(/^---\n[\s\S]*?\n---\n*/, '');
    const lines = cleaned.split('\n');
    const parsed = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i += 1;
        continue;
      }
      if (line.startsWith('```')) {
        const lang = line.slice(3).trim();
        const codeLines = [];
        i += 1;
        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i += 1;
        }
        if (i < lines.length) i += 1;
        parsed.push({ type: 'code', lang, content: codeLines.join('\n') });
        continue;
      }
      const heading = line.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        parsed.push({ type: 'heading', level: heading[1].length, content: heading[2].trim() });
        i += 1;
        continue;
      }
      if (/^[-*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^[-*]\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^[-*]\s+/, '').trim());
          i += 1;
        }
        parsed.push({ type: 'ul', items });
        continue;
      }
      if (/^\d+\.\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\d+\.\s+/, '').trim());
          i += 1;
        }
        parsed.push({ type: 'ol', items });
        continue;
      }
      if (/^>\s+/.test(line)) {
        const quoteLines = [];
        while (i < lines.length && /^>\s+/.test(lines[i])) {
          quoteLines.push(lines[i].replace(/^>\s+/, ''));
          i += 1;
        }
        parsed.push({ type: 'quote', content: quoteLines.join(' ') });
        continue;
      }
      const paraLines = [];
      while (
        i < lines.length &&
        lines[i].trim() &&
        !lines[i].startsWith('```') &&
        !/^(#{1,4})\s+/.test(lines[i]) &&
        !/^[-*]\s+/.test(lines[i]) &&
        !/^\d+\.\s+/.test(lines[i]) &&
        !/^>\s+/.test(lines[i])
      ) {
        paraLines.push(lines[i].trim());
        i += 1;
      }
      parsed.push({ type: 'p', content: paraLines.join(' ') });
    }
    return parsed;
  }, [source]);

  return (
    <div className="space-y-5">
      {blocks.map((block, idx) => {
        if (block.type === 'heading') {
          const cls =
            block.level === 1 ? 'text-2xl font-bold text-zinc-900' :
            block.level === 2 ? 'text-xl font-bold text-zinc-900' :
            block.level === 3 ? 'text-lg font-semibold text-zinc-900' :
            'text-sm font-semibold text-zinc-800 uppercase tracking-wide';
          const Tag = block.level <= 2 ? `h${block.level + 1}` : 'h4';
          return <Tag key={idx} className={cls}>{renderInlineMarkdown(block.content, `h-${idx}`)}</Tag>;
        }
        if (block.type === 'code') {
          return <CodeBlock key={idx} lang={block.lang}>{block.content}</CodeBlock>;
        }
        if (block.type === 'ul') {
          return (
            <ul key={idx} className="space-y-2 text-zinc-600 pl-5 list-disc">
              {block.items.map((item, itemIdx) => (
                <li key={itemIdx}>{renderInlineMarkdown(item, `ul-${idx}-${itemIdx}`)}</li>
              ))}
            </ul>
          );
        }
        if (block.type === 'ol') {
          return (
            <ol key={idx} className="space-y-2 text-zinc-600 pl-5 list-decimal">
              {block.items.map((item, itemIdx) => (
                <li key={itemIdx}>{renderInlineMarkdown(item, `ol-${idx}-${itemIdx}`)}</li>
              ))}
            </ol>
          );
        }
        if (block.type === 'quote') {
          return (
            <blockquote key={idx} className="border-l-4 border-zinc-200 pl-4 text-zinc-600 italic">
              {renderInlineMarkdown(block.content, `q-${idx}`)}
            </blockquote>
          );
        }
        return (
          <p key={idx} className="text-zinc-600 leading-relaxed">
            {renderInlineMarkdown(block.content, `p-${idx}`)}
          </p>
        );
      })}
    </div>
  );
}

// ==========================================
// 🧭 NAVIGATION
// ==========================================

const Navbar = ({ active, setActive }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => { const fn = () => setScrolled(window.scrollY > 10); window.addEventListener('scroll', fn); return () => window.removeEventListener('scroll', fn); }, []);

  const tabs = [{ id: 'home', label: 'Home' }, { id: 'leaderboard', label: 'Leaderboard' }, { id: 'docs', label: 'Docs' }];
  return (
    <nav className={cn("sticky top-0 z-50 transition-all duration-200", scrolled ? "bg-white/95 backdrop-blur-lg border-b border-zinc-200 shadow-sm" : "bg-white border-b border-zinc-100")}>
      <div className="max-w-6xl mx-auto px-5 flex justify-between items-center h-14">
        <button onClick={() => setActive('home')} className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-600 to-red-500 flex items-center justify-center"><Shield className="text-white" size={14} /></div>
          <span className="text-lg font-bold tracking-tight"><span className="text-blue-700">PI</span><span className="text-red-600">Arena</span></span>
        </button>
        <div className="hidden md:flex items-center gap-1">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setActive(t.id)}
              className={cn("px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors", active === t.id ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50")}>{t.label}</button>
          ))}
          <div className="w-px h-5 bg-zinc-200 mx-2" />
          <a href={PROJECT.github} target="_blank" rel="noopener noreferrer" className="p-2 text-zinc-400 hover:text-zinc-900 transition-colors"><Github size={18} /></a>
          <a href={PROJECT.huggingface} target="_blank" rel="noopener noreferrer" className="p-2 text-zinc-400 hover:text-amber-500 transition-colors"><Database size={18} /></a>
        </div>
        <button onClick={() => setMobileOpen(!mobileOpen)} className="md:hidden p-2 text-zinc-500">{mobileOpen ? <X size={20} /> : <Menu size={20} />}</button>
      </div>
      {mobileOpen && (
        <div className="md:hidden border-t border-zinc-100 bg-white px-5 py-3 space-y-1">
          {tabs.map(t => <button key={t.id} onClick={() => { setActive(t.id); setMobileOpen(false); }} className={cn("block w-full text-left px-3 py-2 rounded-lg text-sm font-medium", active === t.id ? "bg-zinc-100 text-zinc-900" : "text-zinc-500")}>{t.label}</button>)}
        </div>
      )}
    </nav>
  );
};

// ==========================================
// 🏠 HOME
// ==========================================

const HomePage = ({ setActive }) => (
  <div>
    {/* Hero */}
    <section className="relative pt-20 pb-20 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-blue-50/50 via-white to-white" />
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-[350px] bg-gradient-to-r from-blue-100/30 via-red-100/20 to-blue-100/30 rounded-full blur-3xl" />
      <div className="relative max-w-6xl mx-auto px-5 text-center">
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-4">
          <span className="text-blue-700">PI</span><span className="text-red-600">Arena</span>
        </h1>
        <p className="text-xl md:text-2xl font-semibold text-zinc-700 mb-4">A Platform for Prompt Injection Evaluation</p>
        <p className="max-w-2xl mx-auto text-zinc-500 leading-relaxed">
          Enabling plug-and-play integration and systematic evaluation of prompt injection attacks and defenses for LLMs and Agents.
        </p>
        <div className="mt-10 flex flex-wrap justify-center gap-3">
          <button onClick={() => setActive('leaderboard')} className="inline-flex items-center gap-2 px-6 py-3 bg-zinc-900 text-white rounded-xl font-semibold text-sm hover:bg-zinc-800 transition-colors shadow-md"><Trophy size={16} /> Leaderboard</button>
          <button onClick={() => setActive('docs')} className="inline-flex items-center gap-2 px-6 py-3 bg-white text-zinc-700 rounded-xl font-semibold text-sm border border-zinc-200 hover:bg-zinc-50 transition-colors shadow-sm"><Terminal size={16} /> Get Started</button>
          <a href={PROJECT.github} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-6 py-3 bg-white text-zinc-700 rounded-xl font-semibold text-sm border border-zinc-200 hover:bg-zinc-50 transition-colors shadow-sm"><Github size={16} /> GitHub</a>
          <a href={PROJECT.huggingface} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-6 py-3 bg-white text-zinc-700 rounded-xl font-semibold text-sm border border-zinc-200 hover:bg-zinc-50 transition-colors shadow-sm"><Database size={16} /> HuggingFace</a>
          {PROJECT.paper && <a href={PROJECT.paper} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-6 py-3 bg-white text-zinc-700 rounded-xl font-semibold text-sm border border-zinc-200 hover:bg-zinc-50 transition-colors shadow-sm"><FileText size={16} /> Paper</a>}
        </div>
      </div>
    </section>

    {/* Key Findings */}
    <section className="py-16 bg-white border-t border-zinc-100">
      <div className="max-w-6xl mx-auto px-5">
        <div className="mb-10">
          <h2 className="text-3xl font-bold text-zinc-900 tracking-tight">What We Discovered</h2>
          <p className="mt-3 text-lg text-zinc-500">Through systematic evaluation using PIArena, we uncover critical limitations of state-of-the-art prompt injection defenses.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            { icon: <AlertTriangle size={20} />, c: "text-rose-600 bg-rose-50 border-rose-100", title: "Limited Generalizability", d: "State-of-the-art defenses may perform well on specific tasks but fail to transfer to other benchmarks and settings." },
            { icon: <Zap size={20} />, c: "text-amber-600 bg-amber-50 border-amber-100", title: "Adaptive Attacks Bypass Defenses", d: "We design an adaptive, strategy-based attack based on defense feedbacks, bypassing existing defenses." },
            { icon: <Cpu size={20} />, c: "text-blue-600 bg-blue-50 border-blue-100", title: "Closed-source LLMs Still Vulnerable", d: "GPT-5, Claude-Sonnet-4.5, and Gemini-3-Pro all exhibit high ASRs under prompt injection, even with built-in defenses." },
            { icon: <Target size={20} />, c: "text-violet-600 bg-violet-50 border-violet-100", title: "Aligned Tasks Fundamentally Hard", d: "When injected tasks align with the target task, attacks can reduce to disinformation—making defense fundamentally challenging." },
          ].map((f, i) => (
            <div key={i} className={cn("flex gap-4 p-5 rounded-xl border", f.c.split(' ').slice(1).join(' '))}>
              <div className={cn("flex-shrink-0 mt-0.5", f.c.split(' ')[0])}>{f.icon}</div>
              <div>
                <h3 className="font-semibold text-zinc-900 mb-1">{f.title}</h3>
                <p className="text-sm text-zinc-600 leading-relaxed">{f.d}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>

    {/* Realistic Injected Tasks */}
    <section className="py-16 bg-zinc-50 border-y border-zinc-200">
      <div className="max-w-6xl mx-auto px-5">
        <div className="mb-10">
          <h2 className="text-3xl font-bold text-zinc-900 tracking-tight">Realistic Injected Tasks</h2>
          <p className="mt-3 text-lg text-zinc-500">PIArena provides realistic injected tasks representing real-world attacker goals.</p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            { icon: <Link size={20} />, color: "text-rose-600", bg: "bg-rose-50", title: "Phishing Injection", d: "Inject phishing links or redirect users to malicious external websites." },
            { icon: <BarChart size={20} />, color: "text-amber-600", bg: "bg-amber-50", title: "Content Promotion", d: "Embed ads or promotional content recommending specific products." },
            { icon: <AlertTriangle size={20} />, color: "text-orange-600", bg: "bg-orange-50", title: "Access Denial", d: "Block user access by falsely claiming quota exhaustion or expired subscriptions." },
            { icon: <Server size={20} />, color: "text-violet-600", bg: "bg-violet-50", title: "Infra. Failure", d: "Mimic backend failures (OOM, timeouts, HTTP errors) to undermine trust." },
          ].map((t, i) => (
            <div key={i} className="rounded-xl border border-zinc-200 p-5 bg-white">
              <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center mb-3", t.bg, t.color)}>{t.icon}</div>
              <h3 className="font-semibold text-zinc-900 mb-1.5 text-sm">{t.title}</h3>
              <p className="text-xs text-zinc-500 leading-relaxed">{t.d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>

    {/* CTA */}
    <section className="py-16 bg-zinc-900">
      <div className="max-w-6xl mx-auto px-5 text-center">
        <h2 className="text-2xl font-bold text-white mb-3">Ready to evaluate your defenses?</h2>
        <p className="text-zinc-400 mb-8 max-w-lg mx-auto">PIArena is open-source. Start benchmarking in minutes.</p>
        <div className="flex flex-wrap justify-center gap-3">
          <a href={PROJECT.github} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-6 py-3 bg-white text-zinc-900 rounded-xl font-semibold text-sm hover:bg-zinc-100"><Github size={16} /> View on GitHub</a>
          <a href={PROJECT.huggingface} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 px-6 py-3 bg-zinc-800 text-white rounded-xl font-semibold text-sm border border-zinc-700 hover:bg-zinc-700"><Database size={16} /> HuggingFace Datasets</a>
        </div>
      </div>
    </section>
  </div>
);

// ==========================================
// 🏆 LEADERBOARD
// ==========================================

const LeaderboardPage = () => {
  const [view, setView] = useState('defense');
  const [sortKey, setSortKey] = useState('utility');
  const [sortDir, setSortDir] = useState('desc');

  const normalizedResults = useMemo(() => normalizeResults(RESULTS), []);
  const index = useMemo(() => buildIndex(normalizedResults), [normalizedResults]);
  const defenseBoard = useMemo(() => computeDefenseLeaderboard(index, METADATA), [index]);
  const llmBoard = useMemo(() => computeLLMLeaderboard(index, METADATA), [index]);
  const agentBoard = useMemo(() => computeAgentLeaderboard(index, METADATA), [index]);
  const coreDatasetCount = useMemo(() => getCoreDatasetKeys(METADATA).length, []);

  const toggleSort = (key) => { if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc'); else { setSortKey(key); setSortDir('asc'); } };

  const SortTh = ({ label, field, sub }) => (
    <th className="px-4 py-3 text-left cursor-pointer select-none group" onClick={() => toggleSort(field)}>
      <div className="flex items-center gap-1">
        <div>
          <span className="font-semibold text-zinc-500 text-xs uppercase tracking-wider">{label}</span>
          {sub && <div className="text-[10px] text-zinc-400 font-normal normal-case">{sub}</div>}
        </div>
        <ArrowUpDown size={11} className={cn("text-zinc-300 group-hover:text-zinc-500", sortKey === field && "text-zinc-600")} />
      </div>
    </th>
  );

  // Defense sort
  const sortedDefense = useMemo(() => {
    const getVal = (row) => {
      if (sortKey === 'utility') return row.byAttack?.none?.utility ?? -1;
      if (sortKey === 'direct_asr') return row.byAttack?.direct?.asr ?? 999;
      if (sortKey === 'combined_asr') return row.byAttack?.combined?.asr ?? 999;
      if (sortKey === 'strategy_asr') return row.byAttack?.strategy?.asr ?? 999;
      return 0;
    };
    return [...defenseBoard].sort((a, b) => sortDir === 'asc' ? getVal(a) - getVal(b) : getVal(b) - getVal(a));
  }, [defenseBoard, sortKey, sortDir]);

  const sortedLLM = useMemo(() => {
    const getVal = (row) => sortKey === 'asrDirect' ? (row.asrDirect ?? 999) : (row.utilNoAtk ?? -1);
    return [...llmBoard].sort((a, b) => sortDir === 'asc' ? getVal(a) - getVal(b) : getVal(b) - getVal(a));
  }, [llmBoard, sortKey, sortDir]);

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <div className="mb-10">
        <h2 className="text-3xl font-bold text-zinc-900 tracking-tight">Evaluation Results</h2>
        <p className="mt-3 text-lg text-zinc-500 max-w-3xl">Systematic assessment of defense mechanisms and LLMs. All defense results averaged across {coreDatasetCount} datasets.</p>
      </div>

      <div className="flex gap-1 bg-zinc-100 p-1 rounded-xl w-fit mb-8">
        {[
          { id: 'defense', label: 'Defense Robustness', icon: <Shield size={14} />, defSort: 'utility', defDir: 'desc' },
          { id: 'llm', label: 'LLM Vulnerability', icon: <Cpu size={14} />, defSort: 'asrDirect', defDir: 'asc' },
          { id: 'agent', label: 'Agent & External', icon: <Box size={14} />, defSort: null },
        ].map(t => (
          <button key={t.id} onClick={() => { setView(t.id); if (t.defSort) { setSortKey(t.defSort); setSortDir(t.defDir || 'asc'); }}}
            className={cn("inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all", view === t.id ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700")}>
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {/* Defense Table */}
      {view === 'defense' && (
        <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-zinc-100 bg-zinc-50/50 flex flex-wrap justify-between items-center gap-2">
            <div>
              <h3 className="font-bold text-zinc-900 text-sm">State-of-the-Art Defenses</h3>
              <p className="text-xs text-zinc-500">Averaged across {coreDatasetCount} datasets · Qwen3-4B-Instruct backend</p>
            </div>
            <div className="flex gap-4 text-[10px] font-medium text-zinc-400">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" /> Good</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400" /> Medium</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-500" /> Poor</span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-zinc-100 bg-zinc-50/30">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-zinc-500 text-xs uppercase tracking-wider">Defense</th>
                  <th className="px-4 py-3 text-left font-semibold text-zinc-500 text-xs uppercase tracking-wider">Type</th>
                  <SortTh label="Utility" field="utility" sub="No Attack ↑" />
                  <SortTh label="Direct ASR" field="direct_asr" sub="↓ Better" />
                  <SortTh label="Combined ASR" field="combined_asr" sub="↓ Better" />
                  <SortTh label="Strategy ASR" field="strategy_asr" sub="Adaptive ↓" />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-50">
                {sortedDefense.map(row => (
                  <tr key={row.id} className={cn("hover:bg-zinc-50/50 transition-colors", row.id === 'none' && "bg-zinc-50/50")}>
                    <td className="px-4 py-3 font-semibold text-zinc-900">{row.name}</td>
                    <td className="px-4 py-3"><Badge variant={safeLower(row.type)}>{row.type || 'Unknown'}</Badge></td>
                    <td className="px-4 py-3"><AsrCell value={row.byAttack?.none?.utility} inverse /></td>
                    <td className="px-4 py-3"><AsrCell value={row.byAttack?.direct?.asr} /></td>
                    <td className="px-4 py-3"><AsrCell value={row.byAttack?.combined?.asr} /></td>
                    <td className="px-4 py-3"><AsrCell value={row.byAttack?.strategy?.asr} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* LLM Table */}
      {view === 'llm' && (
        <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden max-w-3xl">
          <div className="px-5 py-4 border-b border-zinc-100 bg-zinc-50/50">
            <h3 className="font-bold text-zinc-900 text-sm">LLM Vulnerability (SQuAD v2, No External Defense)</h3>
            <p className="text-xs text-zinc-500">Direct prompt injection ASR across different backend LLMs</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-zinc-100 bg-zinc-50/30">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-zinc-500 text-xs uppercase tracking-wider">Backend LLM</th>
                  <th className="px-4 py-3 text-left font-semibold text-zinc-500 text-xs uppercase tracking-wider">Access</th>
                  <SortTh label="Direct ASR" field="asrDirect" sub="↓ Better" />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-50">
                {sortedLLM.map(row => (
                  <tr key={row.id} className="hover:bg-zinc-50/50 transition-colors">
                    <td className="px-4 py-3 font-semibold text-zinc-900">{row.name}</td>
                    <td className="px-4 py-3"><Badge variant={safeLower(row.access)}>{row.access || 'Unknown'}</Badge></td>
                    <td className="px-4 py-3"><AsrCell value={row.asrDirect} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Agent Table */}
      {view === 'agent' && (
        <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden">
          <div className="px-5 py-4 border-b border-zinc-100 bg-zinc-50/50">
            <h3 className="font-bold text-zinc-900 text-sm">Defense on External Benchmarks</h3>
            <p className="text-xs text-zinc-500">PIArena enables plug-and-play defense integration into existing benchmarks</p>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50/30">
                <tr className="border-b border-zinc-100">
                  <th className="px-4 py-2 text-left font-semibold text-zinc-500 text-xs uppercase" rowSpan={2}>Defense</th>
                  <th className="px-4 py-2 text-center font-semibold text-zinc-500 text-xs uppercase border-l border-zinc-100" colSpan={2}>WASP (Agent)</th>
                  <th className="px-4 py-2 text-center font-semibold text-zinc-500 text-xs uppercase border-l border-zinc-100" colSpan={2}>OPI</th>
                  <th className="px-4 py-2 text-center font-semibold text-zinc-500 text-xs uppercase border-l border-zinc-100" colSpan={2}>SEP</th>
                </tr>
                <tr className="border-b border-zinc-100">
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400 border-l border-zinc-100">Util ↑</th>
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400">ASR ↓</th>
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400 border-l border-zinc-100">Util ↑</th>
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400">ASR ↓</th>
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400 border-l border-zinc-100">Util ↑</th>
                  <th className="px-3 py-1 text-center text-[10px] text-zinc-400">ASR ↓</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-50">
                {agentBoard.map(row => (
                  <tr key={row.id} className={cn("hover:bg-zinc-50/50", row.id === 'none' && "bg-zinc-50/50")}>
                    <td className="px-4 py-2.5 font-semibold text-zinc-900">{row.name}</td>
                    {['wasp', 'opi', 'sep'].map(ds => (
                      <React.Fragment key={ds}>
                        <td className="px-3 py-2.5 text-center font-mono text-xs text-zinc-600 border-l border-zinc-50">{row[`${ds}_util`] != null ? `${row[`${ds}_util`]}%` : '—'}</td>
                        <td className="px-3 py-2.5 text-center"><AsrCell value={row[`${ds}_asr`]} /></td>
                      </React.Fragment>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

// ==========================================
// 📖 DOCS
// ==========================================

const DOCS_SECTIONS = [
  { id: 'getting-started', title: 'Getting Started', icon: <Terminal size={16} /> },
  { id: 'attacks-and-defenses', title: 'Attacks & Defenses', icon: <Zap size={16} /> },
  { id: 'evaluation', title: 'Evaluation Pipeline', icon: <Layers size={16} /> },
  { id: 'strategy-search', title: 'Strategy Search', icon: <Crosshair size={16} /> },
  { id: 'agent-benchmarks', title: 'Agent Benchmarks', icon: <Box size={16} /> },
  { id: 'your-own', title: 'Build Your Own', icon: <Code size={16} /> },
];

const DOCS_MARKDOWN = {
  'getting-started': GETTING_STARTED_MD,
  'attacks-and-defenses': ATTACKS_AND_DEFENSES_MD,
  evaluation: EVALUATION_MD,
  'strategy-search': STRATEGY_SEARCH_MD,
  'agent-benchmarks': AGENT_BENCHMARKS_MD,
  'your-own': YOUR_OWN_MD,
};

const DocsContent = ({ section }) => <MarkdownDoc source={DOCS_MARKDOWN[section] || 'No documentation found.'} />;

const DOCS_IDS = DOCS_SECTIONS.map(s => s.id);

const DocsPage = () => {
  const { subPath } = parseHash();
  const initialSec = DOCS_IDS.includes(subPath) ? subPath : 'getting-started';
  const [sec, setSec] = useState(initialSec);

  useEffect(() => {
    const onHashChange = () => {
      const { tab, subPath: sp } = parseHash();
      if (tab === 'docs' && DOCS_IDS.includes(sp)) setSec(sp);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);
  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <div className="flex flex-col md:flex-row gap-6">
        <div className="w-full md:w-52 flex-shrink-0">
          <div className="md:sticky md:top-20 bg-white rounded-xl border border-zinc-200 p-3">
            <nav className="space-y-0.5">
              {DOCS_SECTIONS.map(s => (
                <button key={s.id} onClick={() => { setSec(s.id); window.location.hash = `#/docs/${s.id}`; }}
                  className={cn("w-full flex items-center gap-2.5 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors text-left",
                    sec === s.id ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700")}>
                  <span className={cn(sec === s.id ? "text-zinc-700" : "text-zinc-400")}>{s.icon}</span>{s.title}
                </button>
              ))}
            </nav>
          </div>
        </div>
        <div className="flex-1 min-w-0 bg-white rounded-xl border border-zinc-200">
          <div className="px-6 py-6"><DocsContent section={sec} /></div>
        </div>
      </div>
    </div>
  );
};

// ==========================================
// 🦶 FOOTER
// ==========================================

const Footer = () => (
  <footer className="bg-white border-t border-zinc-200 py-8 mt-auto">
    <div className="max-w-6xl mx-auto px-5 flex flex-col md:flex-row justify-between items-center gap-4">
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-600 to-red-500 flex items-center justify-center"><Shield className="text-white" size={12} /></div>
        <span className="font-bold text-sm"><span className="text-blue-700">PI</span><span className="text-red-600">Arena</span></span>
        <span className="text-zinc-400 text-xs ml-2">Unified prompt injection evaluation</span>
      </div>
      <div className="flex items-center gap-4">
        <a href={PROJECT.github} target="_blank" rel="noopener noreferrer" className="text-zinc-400 hover:text-zinc-900"><Github size={18} /></a>
        <a href={PROJECT.huggingface} target="_blank" rel="noopener noreferrer" className="text-zinc-400 hover:text-amber-500"><Database size={18} /></a>
        {PROJECT.paper && <a href={PROJECT.paper} target="_blank" rel="noopener noreferrer" className="text-zinc-400 hover:text-zinc-900"><FileText size={18} /></a>}
      </div>
    </div>
    <div className="mt-4 text-center text-xs text-zinc-400">© {new Date().getFullYear()} PIArena Contributors</div>
  </footer>
);

// ==========================================
// 🚀 APP
// ==========================================

const VALID_TABS = ['home', 'leaderboard', 'docs'];

function parseHash() {
  const raw = window.location.hash.replace(/^#\/?/, '');
  const [tab, ...rest] = raw.split('/');
  const subPath = rest.join('/') || null;
  return { tab: VALID_TABS.includes(tab) ? tab : 'home', subPath };
}

function getTabFromHash() {
  return parseHash().tab;
}

export default function App() {
  const [tab, setTab] = useState(getTabFromHash);
  const isFirstRender = React.useRef(true);

  useEffect(() => {
    const onHashChange = () => setTab(getTabFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
    } else {
      window.location.hash = tab === 'home' ? '' : `#/${tab}`;
    }
    window.scrollTo(0, 0);
  }, [tab]);
  return (
    <div className="min-h-screen flex flex-col bg-white" style={{ fontFamily: "'DM Sans', -apple-system, sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      <Navbar active={tab} setActive={setTab} />
      <main className="flex-1">
        <PageErrorBoundary key={tab} page={tab} onHome={() => setTab('home')}>
          {tab === 'home' && <HomePage setActive={setTab} />}
          {tab === 'leaderboard' && <LeaderboardPage />}
          {tab === 'docs' && <DocsPage />}
        </PageErrorBoundary>
      </main>
      <Footer />
    </div>
  );
}
