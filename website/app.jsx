import React, { useState, useEffect, useMemo, useCallback } from 'react';
import METADATA from './data/metadata.json';
import RESULTS from './data/results.json';
import {
  Github, Trophy, Shield, Terminal, Database, Layers, ArrowRight,
  AlertTriangle, CheckCircle, Crosshair, Code, BarChart, Server, Link,
  ChevronRight, ExternalLink, Copy, Zap, ArrowUpDown, Menu, X,
  FileText, Box, Target, Cpu, Filter, ChevronDown
} from 'lucide-react';
import {
  BarChart as RechartsBarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
  ScatterChart, Scatter, ZAxis, Cell as RechartsCell, ReferenceLine, LabelList
} from 'recharts';

// ==========================================
// 📊 DATA LAYER
// ==========================================

const PROJECT = {
  github: "https://github.com/sleeepeer/PIArena",
  huggingface: "https://huggingface.co/datasets/sleeepeer/PIArena",
  paper: null,
};

const DOC_FILES = import.meta.glob('../docs/**/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
});

const TOP_DOC_ORDER = ['getting-started', 'evaluation', 'attacks', 'defenses', 'extending'];
const ATTACK_DOC_ORDER = ['none', 'direct', 'ignore', 'completion', 'character', 'combined', 'nanogcg', 'pair', 'tap', 'strategy-search'];
const DEFENSE_DOC_ORDER = ['none', 'pisanitizer', 'secalign', 'datasentinel', 'attentiontracker', 'promptguard', 'promptlocate', 'promptarmor', 'datafilter', 'piguard'];
const DOC_REDIRECTS = {
  'attacks-and-defenses': 'attacks',
  'strategy-search': 'attacks/strategy-search',
  'agent-benchmarks': 'evaluation',
  'your-own': 'extending',
  'your-own-attacks-defenses': 'extending',
};

function humanizeSlugPart(part) {
  return String(part || '')
    .split(/[-_]/g)
    .filter(Boolean)
    .map(token => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ');
}

function parseDocFrontmatter(raw) {
  const match = String(raw || '').match(/^---\n([\s\S]*?)\n---\n*/);
  if (!match) return {};
  const data = {};
  match[1].split('\n').forEach(line => {
    const m = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
    if (!m) return;
    data[m[1]] = m[2].replace(/^['"]|['"]$/g, '').trim();
  });
  return data;
}

function stripDocFrontmatter(raw) {
  return String(raw || '').replace(/^---\n[\s\S]*?\n---\n*/, '');
}

function slugFromDocPath(path) {
  const relative = String(path || '')
    .replace(/^\.\.\/docs\//, '')
    .replace(/\.md$/, '');
  return relative.endsWith('/index') ? relative.slice(0, -'/index'.length) : relative;
}

function docSortIndex(slug) {
  const top = slug.split('/')[0];
  if (!slug.includes('/')) {
    const index = TOP_DOC_ORDER.indexOf(slug);
    return index === -1 ? 999 : index;
  }
  if (top === 'attacks') {
    const child = slug.split('/')[1];
    const index = child === undefined ? -1 : ATTACK_DOC_ORDER.indexOf(child);
    return index === -1 ? 999 : index;
  }
  if (top === 'defenses') {
    const child = slug.split('/')[1];
    const index = child === undefined ? -1 : DEFENSE_DOC_ORDER.indexOf(child);
    return index === -1 ? 999 : index;
  }
  return 999;
}

function compareDocs(a, b) {
  const aParts = a.slug.split('/');
  const bParts = b.slug.split('/');
  if (aParts.length !== bParts.length) return aParts.length - bParts.length;
  const aIndex = docSortIndex(a.slug);
  const bIndex = docSortIndex(b.slug);
  if (aIndex !== bIndex) return aIndex - bIndex;
  return a.slug.localeCompare(b.slug);
}

function buildDocsIndex() {
  return Object.entries(DOC_FILES)
    .map(([path, raw]) => {
      const frontmatter = parseDocFrontmatter(raw);
      const slug = frontmatter.slug || slugFromDocPath(path);
      return {
        path,
        slug,
        title: frontmatter.title || humanizeSlugPart(slug.split('/').slice(-1)[0]),
        category: frontmatter.category || (slug.includes('/') ? slug.split('/')[0] : 'guide'),
        source: stripDocFrontmatter(raw),
        isIndex: /\/index\.md$/.test(path),
        depth: slug.split('/').length,
      };
    })
    .sort(compareDocs);
}

const DOCS = buildDocsIndex();
const DOCS_BY_SLUG = Object.fromEntries(DOCS.map(doc => [doc.slug, doc]));
const DOC_ROOT_PAGES = DOCS.filter(doc => doc.depth === 1);
const DOC_GROUPS = DOCS
  .filter(doc => doc.isIndex && doc.depth === 1 && (doc.slug === 'attacks' || doc.slug === 'defenses'))
  .map(group => ({
    ...group,
    children: DOCS.filter(doc => doc.slug.startsWith(`${group.slug}/`) && doc.depth === 2),
  }));

function resolveDocSlug(slug) {
  const cleaned = String(slug || '')
    .replace(/^\/+/, '')
    .replace(/^docs\//, '')
    .replace(/\.md$/, '')
    .replace(/\/index$/, '');
  const redirected = DOC_REDIRECTS[cleaned] || cleaned;
  return DOCS_BY_SLUG[redirected] ? redirected : 'getting-started';
}

function docsHrefFromSlug(slug) {
  return `#/docs/${resolveDocSlug(slug)}`;
}

function resolveInlineHref(href) {
  if (!href) return { href: '#', external: false };
  if (/^(https?:|mailto:)/i.test(href)) return { href, external: true };
  if (href.startsWith('#/docs/')) return { href, external: false };
  if (href.startsWith('/docs/')) return { href: `#${href}`, external: false };
  if (href.startsWith('/')) return { href, external: false };

  const normalized = href
    .replace(/^\.\//, '')
    .replace(/^\//, '')
    .replace(/\.md$/, '')
    .replace(/\/index$/, '');

  if (DOCS_BY_SLUG[normalized]) {
    return { href: docsHrefFromSlug(normalized), external: false };
  }

  return { href, external: true };
}

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

const QA_DATASETS = ['squad_v2', 'dolly_qa', 'dolly_ie', 'dolly_summ'];
const RAG_DATASETS = ['nq_rag', 'msmarco_rag', 'hotpotqa_rag'];

function getDatasetKeysByGroup(meta, group) {
  if (group === 'all') {
    return Object.keys(meta.datasets).filter(d => {
      const g = meta.datasets[d]?.group;
      return g === 'short' || g === 'long';
    });
  }
  return Object.keys(meta.datasets).filter(d => meta.datasets[d]?.group === group);
}

function getCoreDatasetKeys(meta) {
  return getDatasetKeysByGroup(meta, 'all');
}

const ATTACK_TYPES = ['none', 'direct', 'combined', 'strategy'];
const ACTIVE_ATTACKS = ['direct', 'combined', 'strategy'];

function computeDefenseLeaderboard(index, meta, { datasets, llm }) {
  const defenseKeys = Object.keys(meta.defenses);

  // Compute no-defense baseline utility (attack=none, defense=none)
  const baselineVals = datasets.map(ds => lookup(index, ds, 'none', 'none', llm)).filter(Boolean);
  const baselineUtil = baselineVals.length > 0
    ? baselineVals.map(v => v.utility).filter(v => v !== null)
    : [];
  const baselineAvgUtil = baselineUtil.length > 0
    ? Math.round(baselineUtil.reduce((a,b) => a+b, 0) / baselineUtil.length)
    : null;

  return defenseKeys.map(defKey => {
    const defMeta = meta.defenses[defKey];
    const byAttack = {};
    ATTACK_TYPES.forEach(atk => {
      const vals = datasets.map(ds => lookup(index, ds, atk, defKey, llm)).filter(Boolean);
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
    // Relative utility: how much utility does this defense preserve vs no-defense baseline
    const cleanUtil = byAttack.none?.utility ?? null;
    const relativeUtil = (cleanUtil !== null && baselineAvgUtil !== null && baselineAvgUtil > 0)
      ? Math.round((cleanUtil / baselineAvgUtil) * 100)
      : null;
    // Average ASR across active attack types
    const atkAsrs = ACTIVE_ATTACKS.map(a => byAttack[a]?.asr).filter(v => v !== null && v !== undefined);
    const avgAsr = atkAsrs.length > 0 ? Math.round(atkAsrs.reduce((a,b) => a+b, 0) / atkAsrs.length) : null;
    return { id: defKey, name: defMeta.display, type: defMeta.type, byAttack, avgAsr, cleanUtil, relativeUtil, baselineAvgUtil };
  });
}

function computeAgentLeaderboard(index, meta) {
  const agentDatasets = Object.keys(meta.datasets).filter(d => {
    const g = meta.datasets[d]?.group;
    return g === 'agent' || g === 'external';
  });
  const defenses = Object.keys(meta.defenses);

  return defenses.map(defKey => {
    const row = { id: defKey, name: meta.defenses[defKey]?.display || defKey, type: meta.defenses[defKey]?.type };
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
  const raw = String(text || '');
  const pattern = /\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`|\*\*([^*]+)\*\*/g;
  let lastIndex = 0;
  let match;
  let partIndex = 0;

  while ((match = pattern.exec(raw)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(
        <React.Fragment key={`${keyPrefix}-txt-${partIndex++}`}>
          {raw.slice(lastIndex, match.index)}
        </React.Fragment>
      );
    }

    if (match[1] !== undefined && match[2] !== undefined) {
      const resolved = resolveInlineHref(match[2]);
      nodes.push(
        <a
          key={`${keyPrefix}-link-${partIndex++}`}
          href={resolved.href}
          target={resolved.external ? "_blank" : undefined}
          rel={resolved.external ? "noopener noreferrer" : undefined}
          className="text-blue-700 hover:text-blue-800 underline underline-offset-2"
        >
          {renderInlineMarkdown(match[1], `${keyPrefix}-label-${partIndex}`)}
        </a>
      );
    } else if (match[3] !== undefined) {
      nodes.push(
        <code key={`${keyPrefix}-code-${partIndex++}`} className="px-1.5 py-0.5 bg-zinc-100 rounded text-zinc-800 text-xs font-mono">
          {match[3]}
        </code>
      );
    } else if (match[4] !== undefined) {
      nodes.push(
        <strong key={`${keyPrefix}-strong-${partIndex++}`} className="font-semibold text-zinc-900">
          {match[4]}
        </strong>
      );
    }

    lastIndex = pattern.lastIndex;
  }

  if (lastIndex < raw.length) {
    nodes.push(
      <React.Fragment key={`${keyPrefix}-txt-${partIndex++}`}>
        {raw.slice(lastIndex)}
      </React.Fragment>
    );
  }

  return nodes;
}

function MarkdownDoc({ source }) {
  const blocks = useMemo(() => {
    const cleaned = stripDocFrontmatter(source).replace(/\r\n/g, '\n');
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

// ── Leaderboard Primitives ──

const DATASET_GROUPS = [
  { id: 'all', label: 'All Datasets' },
  { id: 'short', label: 'Short-Context' },
  { id: 'long', label: 'Long-Context' },
];

const DEFENSE_TYPE_FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'Prevention', label: 'Prevention' },
  { id: 'Detection', label: 'Detection' },
];

const CHART_COLORS = {
  direct: '#fecdd3',      // Rose-200 (shallowest)
  combined: '#fb7185',    // Rose-400 (medium)
  strategy: '#f43f5e',    // Rose-500 (deepest — same as "Poor" in defense table)
  none: '#6ee7b7',
};

const SCATTER_DEFENSE_COLORS = {
  Prevention: '#0ea5e9',
  Detection: '#8b5cf6',
  'No Defense': '#ef4444',
  Baseline: '#a1a1aa',
};

const FilterSelect = ({ label, value, onChange, options, className = '' }) => (
  <div className={cn("flex flex-col gap-1", className)}>
    <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">{label}</label>
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="text-sm font-medium text-zinc-800 bg-white border border-zinc-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 cursor-pointer appearance-none bg-[url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2212%22%20height%3D%2212%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%2371717a%22%20stroke-width%3D%222%22%3E%3Cpath%20d%3D%22M6%209l6%206%206-6%22%2F%3E%3C%2Fsvg%3E')] bg-no-repeat bg-[right_8px_center] pr-7"
    >
      {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  </div>
);

const DATASET_CATEGORIES = [
  { label: 'QA', keys: QA_DATASETS },
  { label: 'RAG', keys: RAG_DATASETS },
  { label: 'Long Context', filter: (meta) => Object.keys(meta.datasets).filter(d => meta.datasets[d]?.group === 'long') },
];

const DatasetChips = ({ meta, group, selected, onToggle, onSelectAll, onDeselectAll }) => {
  const datasets = getDatasetKeysByGroup(meta, group);
  if (datasets.length === 0) return null;
  const allSelected = datasets.every(ds => selected.includes(ds));

  // Build visible categories: only show categories that have datasets in current group
  const categories = DATASET_CATEGORIES
    .map(cat => ({
      label: cat.label,
      items: (cat.keys || cat.filter(meta)).filter(d => datasets.includes(d)),
    }))
    .filter(cat => cat.items.length > 0);

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <button
        onClick={() => allSelected ? onDeselectAll(datasets) : onSelectAll(datasets)}
        className={cn(
          "px-2.5 py-1 rounded-md text-xs font-semibold transition-all border",
          allSelected
            ? "border-zinc-300 bg-zinc-800 text-white"
            : "border-zinc-200 bg-white text-zinc-500 hover:bg-zinc-100"
        )}
      >
        {allSelected ? 'Deselect All' : 'Select All'}
      </button>
      {categories.map((cat, ci) => (
        <React.Fragment key={cat.label}>
          <span className="h-5 mx-1 border-l border-dashed border-zinc-300" />
          <span className="text-[10px] text-zinc-400 font-medium mr-0.5">{cat.label}</span>
          {cat.items.map(ds => {
            const active = selected.includes(ds);
            return (
              <button
                key={ds}
                onClick={() => onToggle(ds)}
                className={cn(
                  "px-2.5 py-1 rounded-md text-xs font-medium transition-all",
                  active
                    ? "bg-zinc-900 text-white shadow-sm"
                    : "bg-zinc-100 text-zinc-500 hover:bg-zinc-200 hover:text-zinc-700"
                )}
              >
                {meta.datasets[ds]?.display || ds}
              </button>
            );
          })}
        </React.Fragment>
      ))}
      <span className="text-[10px] text-zinc-400 ml-1">{selected.filter(s => datasets.includes(s)).length}/{datasets.length}</span>
    </div>
  );
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white rounded-lg border border-zinc-200 shadow-lg px-3 py-2 text-xs">
      <p className="font-semibold text-zinc-900 mb-1">{label}</p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.fill || p.color }} />
          <span className="text-zinc-500">{p.name}:</span>
          <span className="font-mono font-semibold text-zinc-800">{p.value !== null && p.value !== undefined ? `${p.value}%` : '—'}</span>
        </div>
      ))}
    </div>
  );
};

const ScatterTooltipContent = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;
  return (
    <div className="bg-white rounded-lg border border-zinc-200 shadow-lg px-3 py-2 text-xs">
      <p className="font-semibold text-zinc-900 mb-1">{data.name}</p>
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">Utility:</span>
          <span className="font-mono font-semibold text-zinc-800">{data.utility !== null ? `${data.utility}%` : '—'}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">ASR:</span>
          <span className="font-mono font-semibold text-zinc-800">{data.asr !== null ? `${data.asr}%` : '—'}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">Type:</span>
          <span className="font-semibold text-zinc-800">{data.type}</span>
        </div>
      </div>
    </div>
  );
};


// ── Main Leaderboard ──

const LeaderboardPage = () => {
  const [sortKey, setSortKey] = useState('combined_asr');
  const [sortDir, setSortDir] = useState('asc');

  // Filter state
  const [datasetGroup, setDatasetGroup] = useState('all');
  const [selectedDatasets, setSelectedDatasets] = useState(() => getDatasetKeysByGroup(METADATA, 'all'));
  const [defenseTypeFilter, setDefenseTypeFilter] = useState('all');
  const [scatterAttack, setScatterAttack] = useState('combined');

  const normalizedResults = useMemo(() => normalizeResults(RESULTS), []);
  const index = useMemo(() => buildIndex(normalizedResults), [normalizedResults]);

  // When group changes, reset selected datasets
  const handleGroupChange = useCallback((group) => {
    setDatasetGroup(group);
    setSelectedDatasets(getDatasetKeysByGroup(METADATA, group));
  }, []);

  const handleDatasetToggle = useCallback((ds) => {
    setSelectedDatasets(prev => {
      const next = prev.includes(ds) ? prev.filter(d => d !== ds) : [...prev, ds];
      return next.length > 0 ? next : prev;
    });
  }, []);

  const handleSelectAll = useCallback((datasets) => {
    setSelectedDatasets(prev => [...new Set([...prev, ...datasets])]);
  }, []);

  const handleDeselectAll = useCallback((datasets) => {
    setSelectedDatasets(prev => {
      const next = prev.filter(d => !datasets.includes(d));
      // Keep at least one selected
      return next.length > 0 ? next : [datasets[0]];
    });
  }, []);

  // Computed boards — always qwen3-4b
  const defenseBoard = useMemo(
    () => computeDefenseLeaderboard(index, METADATA, { datasets: selectedDatasets, llm: 'qwen3-4b' }),
    [index, selectedDatasets]
  );
  const agentBoard = useMemo(
    () => computeAgentLeaderboard(index, METADATA),
    [index]
  );

  // Filter defenses by type
  const filteredDefense = useMemo(() => {
    if (defenseTypeFilter === 'all') return defenseBoard;
    return defenseBoard.filter(r => r.type === defenseTypeFilter || r.id === 'none');
  }, [defenseBoard, defenseTypeFilter]);

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

  // Sort
  const sortedDefense = useMemo(() => {
    const getVal = (row) => {
      if (sortKey === 'clean_util') return row.cleanUtil ?? -1;
      if (sortKey === 'direct_asr') return row.byAttack?.direct?.asr ?? 999;
      if (sortKey === 'combined_asr') return row.byAttack?.combined?.asr ?? 999;
      if (sortKey === 'strategy_asr') return row.byAttack?.strategy?.asr ?? 999;
      return 0;
    };
    return [...filteredDefense].sort((a, b) => sortDir === 'asc' ? getVal(a) - getVal(b) : getVal(b) - getVal(a));
  }, [filteredDefense, sortKey, sortDir]);

  // Bar chart data
  const barChartData = useMemo(() => {
    return filteredDefense.map(row => ({
      name: row.name,
      direct: row.byAttack?.direct?.asr ?? null,
      combined: row.byAttack?.combined?.asr ?? null,
      strategy: row.byAttack?.strategy?.asr ?? null,
    }));
  }, [filteredDefense]);

  // Scatter plot data: X = ASR, Y = Utility for the selected attack
  const scatterData = useMemo(() => {
    return filteredDefense
      .map(row => ({
        name: row.name,
        type: row.id === 'none' ? 'No Defense' : (row.type || 'Baseline'),
        asr: row.byAttack?.[scatterAttack]?.asr ?? null,
        utility: row.byAttack?.[scatterAttack]?.utility ?? row.cleanUtil,
      }))
      .filter(d => d.asr !== null && d.utility !== null);
  }, [filteredDefense, scatterAttack]);

  // Agent datasets
  const agentDatasets = useMemo(() =>
    Object.keys(METADATA.datasets).filter(d => {
      const g = METADATA.datasets[d]?.group;
      return g === 'agent' || g === 'external';
    }),
  []);

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-zinc-900 tracking-tight">Evaluation Results</h2>
        <p className="mt-2 text-lg text-zinc-500 max-w-3xl">
          Defense comparison on Qwen3-4B-Instruct across {selectedDatasets.length} dataset{selectedDatasets.length !== 1 ? 's' : ''}.
        </p>
      </div>

      {/* ── Filter Bar ── */}
      <div className="bg-white rounded-xl border border-zinc-200 p-4 mb-6">
        <div className="flex flex-wrap items-end gap-4 mb-3">
          <FilterSelect
            label="Dataset Group"
            value={datasetGroup}
            onChange={handleGroupChange}
            options={DATASET_GROUPS.map(g => ({ value: g.id, label: g.label }))}
          />
          <div className="flex flex-col gap-1">
            <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Defense Type</label>
            <div className="flex gap-1 bg-zinc-100 p-0.5 rounded-lg">
              {DEFENSE_TYPE_FILTERS.map(t => (
                <button key={t.id} onClick={() => setDefenseTypeFilter(t.id)}
                  className={cn("px-3 py-1 rounded-md text-xs font-semibold transition-all", defenseTypeFilter === t.id ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500 hover:text-zinc-700")}>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <DatasetChips meta={METADATA} group={datasetGroup} selected={selectedDatasets} onToggle={handleDatasetToggle} onSelectAll={handleSelectAll} onDeselectAll={handleDeselectAll} />
      </div>

      {/* ── Scatter + Bar side by side ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Scatter Plot: Utility vs ASR */}
        <div className="bg-white rounded-xl border border-zinc-200 p-5">
          <div className="flex justify-between items-start mb-4">
            <div>
              <h3 className="font-bold text-zinc-900 text-sm">Utility vs ASR</h3>
              <p className="text-[10px] text-zinc-400 mt-0.5">Upper-left is better (high utility, low ASR)</p>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Attack</label>
              <select value={scatterAttack} onChange={e => setScatterAttack(e.target.value)}
                className="text-xs font-medium text-zinc-800 bg-white border border-zinc-200 rounded-md px-2 py-1 focus:outline-none cursor-pointer">
                {ACTIVE_ATTACKS.map(a => <option key={a} value={a}>{METADATA.attacks[a]?.display || a}</option>)}
              </select>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
              <XAxis type="number" dataKey="asr" name="ASR" domain={[0, 100]} tick={{ fontSize: 11, fill: '#71717a' }} label={{ value: 'ASR (%) →', position: 'insideBottom', offset: -5, fontSize: 11, fill: '#a1a1aa' }} />
              <YAxis type="number" dataKey="utility" name="Utility" domain={[0, 100]} tick={{ fontSize: 11, fill: '#71717a' }} label={{ value: 'Utility (%)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 11, fill: '#a1a1aa' }} />
              <ZAxis range={[80, 80]} />
              <Tooltip content={<ScatterTooltipContent />} />
              <ReferenceLine x={50} stroke="#e4e4e7" strokeDasharray="3 3" />
              <ReferenceLine y={50} stroke="#e4e4e7" strokeDasharray="3 3" />
              <Scatter data={scatterData} shape="circle">
                {scatterData.map((entry, i) => (
                  <RechartsCell key={i} fill={SCATTER_DEFENSE_COLORS[entry.type] || '#a1a1aa'} />
                ))}
                <LabelList dataKey="name" position="right" offset={6} style={{ fontSize: 9, fill: '#a1a1aa', fontWeight: 500 }} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-4 mt-2 text-[10px] font-medium text-zinc-400">
            {Object.entries(SCATTER_DEFENSE_COLORS).filter(([k]) => k !== 'Baseline').map(([k, c]) => (
              <span key={k} className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />{k}
              </span>
            ))}
          </div>
        </div>

        {/* Bar Chart: ASR by attack */}
        <div className="bg-white rounded-xl border border-zinc-200 p-5">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-zinc-900 text-sm">ASR by Attack Type</h3>
            <div className="flex gap-3 text-[10px] font-medium text-zinc-400">
              {ACTIVE_ATTACKS.map(a => (
                <span key={a} className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full" style={{ background: CHART_COLORS[a] }} />{METADATA.attacks[a]?.display || a}
                </span>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsBarChart data={barChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f4f4f5" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#71717a' }} interval={0} angle={-25} textAnchor="end" height={70} />
              <YAxis tick={{ fontSize: 11, fill: '#71717a' }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="direct" name="Direct" fill={CHART_COLORS.direct} radius={[2,2,0,0]} />
              <Bar dataKey="combined" name="Combined" fill={CHART_COLORS.combined} radius={[2,2,0,0]} />
              <Bar dataKey="strategy" name="Strategy" fill={CHART_COLORS.strategy} radius={[2,2,0,0]} />
            </RechartsBarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Defense Table ── */}
      <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden mb-12">
        <div className="px-5 py-4 border-b border-zinc-100 bg-zinc-50/50 flex flex-wrap justify-between items-center gap-2">
          <div>
            <h3 className="font-bold text-zinc-900 text-sm">Defense Leaderboard</h3>
            <p className="text-xs text-zinc-500">
              {selectedDatasets.length > 1 ? `Averaged across ${selectedDatasets.length} datasets` : METADATA.datasets[selectedDatasets[0]]?.display || selectedDatasets[0]}
              {' · '}Qwen3-4B-Instruct backend
            </p>
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
                <SortTh label="Clean Utility" field="clean_util" sub="No Attack ↑" />
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
                  <td className="px-4 py-3"><AsrCell value={row.cleanUtil} inverse /></td>
                  <td className="px-4 py-3"><AsrCell value={row.byAttack?.direct?.asr} /></td>
                  <td className="px-4 py-3"><AsrCell value={row.byAttack?.combined?.asr} /></td>
                  <td className="px-4 py-3"><AsrCell value={row.byAttack?.strategy?.asr} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Agent & External Benchmarks ── */}
      <div className="mt-8">
        <div className="mb-6">
          <h3 className="text-xl font-bold text-zinc-900 tracking-tight">Agent & External Benchmarks</h3>
          <p className="mt-1 text-sm text-zinc-500">Plug-and-play defense integration into existing agent and external benchmarks.</p>
        </div>
        <div className="bg-white rounded-xl border border-zinc-200 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-zinc-50/30">
                <tr className="border-b border-zinc-100">
                  <th className="px-4 py-2 text-left font-semibold text-zinc-500 text-xs uppercase" rowSpan={2}>Defense</th>
                  <th className="px-4 py-2 text-left font-semibold text-zinc-500 text-xs uppercase" rowSpan={2}>Type</th>
                  {agentDatasets.map(ds => (
                    <th key={ds} className="px-4 py-2 text-center font-semibold text-zinc-500 text-xs uppercase border-l border-zinc-100" colSpan={2}>
                      {METADATA.datasets[ds]?.display || ds}
                    </th>
                  ))}
                </tr>
                <tr className="border-b border-zinc-100">
                  {agentDatasets.map(ds => (
                    <React.Fragment key={ds}>
                      <th className="px-3 py-1 text-center text-[10px] text-zinc-400 border-l border-zinc-100">Utility ↑</th>
                      <th className="px-3 py-1 text-center text-[10px] text-zinc-400">ASR ↓</th>
                    </React.Fragment>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-50">
                {agentBoard.map(row => (
                  <tr key={row.id} className={cn("hover:bg-zinc-50/50", row.id === 'none' && "bg-zinc-50/50")}>
                    <td className="px-4 py-2.5 font-semibold text-zinc-900">{row.name}</td>
                    <td className="px-4 py-2.5"><Badge variant={safeLower(row.type)}>{row.type || 'Unknown'}</Badge></td>
                    {agentDatasets.map(ds => (
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
      </div>
    </div>
  );
};

// ==========================================
// 📖 DOCS
// ==========================================

const DOCS_SECTIONS = [
  { slug: 'getting-started', icon: <Terminal size={16} /> },
  { slug: 'evaluation', icon: <Layers size={16} /> },
  { slug: 'attacks', icon: <Zap size={16} /> },
  { slug: 'defenses', icon: <Shield size={16} /> },
  { slug: 'extending', icon: <Code size={16} /> },
].map(item => ({
  ...item,
  title: DOCS_BY_SLUG[item.slug]?.title || humanizeSlugPart(item.slug),
}));

function DocsContent({ section }) {
  const doc = DOCS_BY_SLUG[section];
  return <MarkdownDoc source={doc?.source || '# Documentation Not Found'} />;
}

const DocsPage = () => {
  const { subPath } = parseHash();
  const initialSec = resolveDocSlug(subPath);
  const [sec, setSec] = useState(initialSec);

  useEffect(() => {
    const onHashChange = () => {
      const { tab, subPath: sp } = parseHash();
      if (tab === 'docs') setSec(resolveDocSlug(sp));
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigateDoc = (slug) => {
    const next = resolveDocSlug(slug);
    setSec(next);
    window.location.hash = `/docs/${next}`;
  };

  const currentGroup = sec.includes('/') ? sec.split('/')[0] : sec;

  return (
    <div className="max-w-6xl mx-auto px-5 py-12">
      <div className="flex flex-col md:flex-row gap-6">
        <div className="w-full md:w-72 flex-shrink-0">
          <div className="md:sticky md:top-20 bg-white rounded-xl border border-zinc-200 p-3">
            <nav className="space-y-0.5">
              {DOCS_SECTIONS.filter(section => section.slug !== 'attacks' && section.slug !== 'defenses').map(section => (
                <button key={section.slug} onClick={() => navigateDoc(section.slug)}
                  className={cn("w-full flex items-center gap-2.5 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors text-left",
                    sec === section.slug ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700")}>
                  <span className={cn(sec === section.slug ? "text-zinc-700" : "text-zinc-400")}>{section.icon}</span>{section.title}
                </button>
              ))}
              <div className="my-2 h-px bg-zinc-100" />
              {DOC_GROUPS.map(group => {
                const groupMeta = DOCS_SECTIONS.find(section => section.slug === group.slug);
                const open = currentGroup === group.slug;
                return (
                  <div key={group.slug} className="space-y-1">
                    <button
                      onClick={() => navigateDoc(group.slug)}
                      className={cn(
                        "w-full flex items-center justify-between gap-2 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors text-left",
                        sec === group.slug ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700"
                      )}
                    >
                      <span className="flex items-center gap-2.5 min-w-0">
                        <span className={cn(sec === group.slug ? "text-zinc-700" : "text-zinc-400")}>{groupMeta?.icon}</span>
                        <span>{group.title}</span>
                      </span>
                      {open ? <ChevronDown size={15} className="text-zinc-400" /> : <ChevronRight size={15} className="text-zinc-400" />}
                    </button>
                    {open && (
                      <div className="pl-3 space-y-0.5">
                        {group.children.map(child => (
                          <button
                            key={child.slug}
                            onClick={() => navigateDoc(child.slug)}
                            className={cn(
                              "w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors text-left",
                              sec === child.slug ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:bg-zinc-50 hover:text-zinc-700"
                            )}
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-zinc-300 flex-shrink-0" />
                            <span>{child.title}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
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
