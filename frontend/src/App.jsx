import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight, Search, Microscope, Brain, GitBranch, Globe, Map, Database, Info, Loader2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const PAGES = [
  { id: 0, title: "Genotype Search", icon: Search },
  { id: 1, title: "Phenotype Discovery", icon: Microscope },
  { id: 2, title: "AI Recommendations", icon: Brain },
  { id: 3, title: "Genotype Analysis", icon: Database },
  { id: 4, title: "Trait Explorer", icon: Globe },
  { id: 5, title: "Knowledge Graphs", icon: GitBranch },
  { id: 6, title: "Map Explorer", icon: Map }
];

function App() {
  const [page, setPage] = useState(0);
  const [direction, setDirection] = useState(0);
  const [genotypes, setGenotypes] = useState([]);
  const [traits, setTraits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const init = async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/init`);
        setGenotypes(res.data.genotypes || []);
        setTraits(res.data.traits || []);
        setLoading(false);
      } catch (err) {
        console.error("Failed to init API", err);
        setLoading(false);
      }
    };
    init();
  }, []);

  const paginate = (newDirection) => {
    let next = page + newDirection;
    if (next >= 0 && next < PAGES.length) {
      setDirection(newDirection);
      setPage(next);
    }
  };

  const variants = {
    enter: (direction) => ({
      x: direction > 0 ? 1000 : -1000,
      opacity: 0,
      scale: 0.8
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1,
      scale: 1
    },
    exit: (direction) => ({
      zIndex: 0,
      x: direction < 0 ? 1000 : -1000,
      opacity: 0,
      scale: 0.8
    }),
  };

  if (loading) return (
    <div className="app-container">
      <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px'}}>
        <Loader2 className="animate-spin" size={60} color="#a8ff78" />
        <h2>Initializing Biological Knowledge Base...</h2>
      </div>
    </div>
  );

  const PageIcon = PAGES[page].icon;

  return (
    <div className="app-container">
      <div className="background-blur"></div>
      
      <header className="main-header">
        <motion.div 
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 260, damping: 20 }}
          style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '15px' }}
        >
          {PageIcon && <PageIcon size={48} color="#a8ff78" />}
          <h1>{PAGES[page].title}</h1>
        </motion.div>
      </header>

      <div className="carousel-wrapper">
        <AnimatePresence initial={false} custom={direction} mode="wait">
          <motion.div
            key={page}
            custom={direction}
            variants={variants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{
              x: { type: "spring", stiffness: 300, damping: 30 },
              opacity: { duration: 0.2 },
              scale: { duration: 0.4 }
            }}
            className="slide-container"
          >
            <StepContent page={page} genotypes={genotypes} traits={traits} />
          </motion.div>
        </AnimatePresence>

        <div className="nav-buttons">
          <button className="nav-btn" onClick={() => paginate(-1)} disabled={page === 0}>
            <ChevronLeft size={32} />
          </button>
          <button className="nav-btn" onClick={() => paginate(1)} disabled={page === PAGES.length - 1}>
            <ChevronRight size={32} />
          </button>
        </div>
      </div>

      <div className="progress-dots">
        {PAGES.map((p, i) => (
          <div 
            key={i} 
            className={`dot ${i === page ? 'active' : ''}`}
            onClick={() => setPage(i)}
          />
        ))}
      </div>
    </div>
  );
}

const StepContent = ({ page, genotypes, traits }) => {
  switch (page) {
    case 0: return <SearchStep genotypes={genotypes} />;
    case 1: return <DiscoveryStep genotypes={genotypes} />;
    case 2: return <AIRecStep />;
    case 3: return <AnalysisStep genotypes={genotypes} />;
    case 4: return <TraitExplorerStep traits={traits} />;
    case 5: return <KnowledgeGraphStep />;
    case 6: return <MapStep genotypes={genotypes} />;
    default: return <div style={{textAlign: 'center', marginTop: '100px'}}><h2>Pro Module Loading</h2><p>Calibrating AI visualization sensors...</p></div>;
  }
};

const SearchStep = ({ genotypes }) => {
  const [q, setQ] = useState("");
  const [res, setRes] = useState([]);

  const onSearch = async (val) => {
    setQ(val);
    if (!val) { setRes([]); return; }
    try {
      const resp = await axios.get(`${API_BASE}/api/search?query=${encodeURIComponent(val)}`);
      setRes(resp.data);
    } catch (err) {}
  };

  return (
    <div className="search-step" style={{display: 'flex', flexDirection: 'column', height: '100%'}}>
      <div style={{position: 'relative', marginBottom: '15px', flexShrink: 0}}>
        <Search style={{position: 'absolute', left: '15px', top: '50%', transform: 'translateY(-50%)', opacity: 0.5}} size={20} />
        <input 
          style={{paddingLeft: '50px', width: '100%', boxSizing: 'border-box'}}
          type="text" 
          placeholder="Search traits, genotypes, states... or type 'high protein', 'drought tolerant', 'tall plant'" 
          value={q}
          onChange={(e) => onSearch(e.target.value)}
        />
      </div>

      {res.length > 0 && (
        <div style={{flexShrink: 0, marginBottom: '10px', color: 'rgba(255,255,255,0.5)', fontSize: '0.8rem'}}>
          📊 {res.length} results for "<strong style={{color: '#a8ff78'}}>{q}</strong>" — scroll to see all
        </div>
      )}

      {/* Scrollable results area */}
      <div style={{flex: 1, overflowY: 'auto', paddingRight: '4px'}}>
        <div className="card-grid">
          {res.map((r, i) => (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.03, 0.3) }}
              key={i} 
              className="metric-card"
            >
              <h3 style={{color: '#a8ff78', fontSize: '1.1rem'}}>{r.Variety || r.Genotype}</h3>
              <p style={{fontSize: '1.7rem'}}>{parseFloat(r.Yield_per_plant).toFixed(2)}kg</p>
              <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '0.85rem', color: '#fff', fontWeight: '600'}}>
                <span>{r.State}</span>
                <span>H: {r.Height}cm</span>
              </div>
              <div style={{marginTop: '6px', fontSize: '0.75rem', color: 'rgba(168,255,120,0.6)'}}>
                {r.Soil_Type} · {r.Drought_Tolerance == 1 ? '💧 Tolerant' : '⚠️ Sensitive'}
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {res.length === 0 && (
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', opacity: 0.8}}>
          <Database size={70} style={{marginBottom: '20px', color: 'var(--primary)'}} />
          <p style={{color: '#fff', fontSize: '1.1rem', textAlign: 'center'}}>
            Search 50,000+ data points<br/>
            <span style={{fontSize: '0.85rem', opacity: 0.6}}>Try: "Kerala" · "high protein" · "drought tolerant" · "clay soil" · "Pusa Basmati"</span>
          </p>
        </div>
      )}
    </div>
  );
};

const DiscoveryStep = ({ genotypes }) => {
  const [sel, setSel] = useState("");
  const [data, setData] = useState(null);

  const onSelect = async (v) => {
    if (!v) return;
    setSel(v);
    try {
      const res = await axios.get(`${API_BASE}/api/genotype/${v}`);
      setData(res.data.data);
    } catch (err) {}
  };

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <select 
        className="custom-select"
        onChange={(e) => onSelect(e.target.value)}
        value={sel}
      >
        <option value="">-- Choose Variety --</option>
        {genotypes.map(g => <option key={g} value={g}>{g}</option>)}
      </select>
      
      {data ? (
        <div className="discovery-results" style={{marginTop: '30px', flex: 1, overflowY: 'auto', paddingRight: '10px'}}>
          <div className="card-grid">
            <MetricCard label="Yield" value={data.Yield_per_plant} unit="kg" />
            <MetricCard label="Height" value={data.Height} unit="cm" />
            <MetricCard label="Grain Weight" value={data.Grain_weight} unit="g" />
            <MetricCard label="Adaptation" value={data.Drought_Tolerance === 1 ? 'TOLERANT' : 'SENSITIVE'} />
          </div>
          <div style={{marginTop: '30px', padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '20px', border: '1px solid var(--glass-border)', marginBottom: '20px'}}>
            <h4 style={{color: '#a8ff78', marginBottom: '10px'}}>ENVIRONMENTAL PROFILE</h4>
            <p style={{opacity: 0.8}}>{data.Variety} is a primary cultivar in {data.State}, {data.Country}. It grows best in {data.Soil_Type} soil with an average temperature of {data.Temperature_C}°C and {data.Rainfall_mm}mm rainfall.</p>
          </div>
        </div>
      ) : (
        <div style={{flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', opacity: 0.8}}>
          <p style={{fontSize: '1.1rem'}}>Select a variety above to unlock deep phenotype metrics</p>
        </div>
      )}
    </div>
  );
};

// Helper: parse raw markdown-style recommendation text into clean JSX
// Enhanced: parse raw text into structured JSX cards/sections
const parseRecText = (text) => {
  if (!text) return null;
  
  const lines = text.split('\n');
  const elements = [];
  let currentCard = null;

  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    // Detect Title (Target/Goal)
    if (line.startsWith('🎯') || line.startsWith('🧠')) {
      elements.push(
        <div key={`title-${i}`} className="rec-section-title">
          {trimmed.replace(/\*\*/g, '')}
        </div>
      );
      return;
    }

    // Detect Start of a Cross Card
    // Matches: "Cross 1: ..." or "**Cross 1: ..."
    const crossMatch = trimmed.match(/^(\*\*)?Cross \d+: (.+?)(\*\*)?$/i);
    if (crossMatch) {
      if (currentCard) elements.push(currentCard);
      
      const title = crossMatch[2].replace(/\*/g, '');
      currentCard = (
        <div key={`card-${i}`} className="suggestion-card">
          <div className="card-header">
            <GitBranch size={18} color="#a8ff78" />
            <span>{title}</span>
          </div>
          <div className="card-body"></div>
        </div>
      );
      return;
    }

    // Detect Knowledge-Based Strategy or Tip (Footer)
    if (line.startsWith('💡') || line.startsWith('🔍') || line.startsWith('🔬')) {
      if (currentCard) {
        elements.push(currentCard);
        currentCard = null;
      }
      elements.push(
        <div key={`footer-${i}`} className="rec-footer-note">
          {trimmed.replace(/\*\*/g, '').replace(/\*/g, '')}
        </div>
      );
      return;
    }

    // Process interior content for cards or standalone lines
    const isStat = trimmed.startsWith('📊') || trimmed.startsWith('📈');
    const isDNA = trimmed.startsWith('🧬') || trimmed.startsWith('🧪');
    const isPoint = trimmed.startsWith('•') || trimmed.startsWith('*') || trimmed.startsWith('-');
    
    const content = trimmed
      .replace(/^[•*\-\s📊📈🧬🧪]+/, '') // remove symbols for clean text
      .replace(/\*\*/g, '');

    const row = (
      <div key={`line-${i}`} className={`rec-row ${isStat ? 'stat' : isDNA ? 'dna' : ''}`}>
        {isStat && <span className="icon">📊</span>}
        {isDNA && <span className="icon">🧬</span>}
        {isPoint && <span className="bullet">•</span>}
        <span className="text">{content}</span>
      </div>
    );

    if (currentCard) {
      // Add to the last created card's body
      const newCard = React.cloneElement(currentCard, {
        children: [
          currentCard.props.children[0], // header
          <div key="body" className="card-body">
            {currentCard.props.children[1].props.children}
            {row}
          </div>
        ]
      });
      currentCard = newCard;
    } else {
      elements.push(row);
    }
  });

  if (currentCard) elements.push(currentCard);
  return elements;
};

const AnalysisStep = ({ genotypes }) => {
  const [sel, setSel] = useState("");
  const [graph, setGraph] = useState(null);
  const [recs, setRecs] = useState("");

  const onSelect = async (v) => {
    if (!v) return;
    setSel(v);
    try {
      const res = await axios.get(`${API_BASE}/api/genotype/${v}`);
      setGraph(res.data.rule_graph);
      setRecs(res.data.recommendations);
    } catch (err) {}
  };

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <select className="custom-select" onChange={(e) => onSelect(e.target.value)} value={sel}>
        <option value="">-- Select Variety for AI Analysis --</option>
        {genotypes.map(g => <option key={g} value={g}>{g}</option>)}
      </select>
      
      <div style={{display: 'flex', flex: 1, gap: '20px', marginTop: '20px', overflow: 'hidden', minHeight: 0}}>
        {/* 3D Rule Graph — full height, scrollable */}
        <div style={{flex: 1.2, background: '#000', borderRadius: '20px', overflowY: 'auto'}}>
          {graph ? (
             <Plot 
               data={graph.data}
               layout={{...graph.layout, autosize: false, width: 520, height: 460,
                 paper_bgcolor: '#000', plot_bgcolor: '#000',
                 margin: {l:0, r:0, b:10, t:40},
                 scene: graph.layout.scene ? {...graph.layout.scene, bgcolor: '#000'} : undefined,
               }}
               style={{width: '100%'}}
             />
          ) : <div style={{height: '420px', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.3}}><GitBranch size={40} /></div>}
        </div>
        {/* Recommendations panel — structured points */}
        <div style={{flex: 1, padding: '20px', background: 'rgba(255,255,255,0.04)', borderRadius: '20px', overflowY: 'auto', fontSize: '0.9rem', border: '1px solid var(--glass-border)'}}>
          <h4 style={{color: '#a8ff78', marginBottom: '15px', borderBottom: '1px solid rgba(168,255,120,0.2)', paddingBottom: '10px'}}>🧬 AI INTELLIGENCE REPORT</h4>
          {recs ? parseRecText(recs) : <p style={{opacity: 0.4}}>Select a variety above to generate the AI report...</p>}
        </div>
      </div>
    </div>
  );
};

const AIRecStep = () => {
  const [goal, setGoal] = useState("");
  const [rec, setRec] = useState("");
  const [kg, setKg] = useState(null);
  const [loading, setLoading] = useState(false);

  const onRecommend = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/api/recommend?goal=${goal}`);
      setRec(res.data.text);
      setKg(res.data.kg_graph);
      setLoading(false);
    } catch (err) { setLoading(false); }
  };

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div style={{display: 'flex', gap: '10px', marginBottom: '20px'}}>
        <input style={{flex: 1}} type="text" placeholder="Enter target (e.g. increase protein and drought tolerance)..." value={goal} onChange={(e) => setGoal(e.target.value)} />
        <button className="gold-btn" onClick={onRecommend} disabled={loading}>
          {loading ? <Loader2 className="animate-spin" /> : 'CORE STRATEGY'}
        </button>
      </div>
      
      <div style={{display: 'flex', flex: 1, gap: '20px', minHeight: 0, overflow: 'hidden'}}>
        {/* Strategy text — structured bullet points */}
        <div style={{flex: 1, padding: '25px', background: 'rgba(168, 255, 120, 0.05)', borderRadius: '25px', overflowY: 'auto', border: '1px solid var(--primary)'}}>
          <h4 style={{color: 'white', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px'}}><Brain /> STRATEGY RECOMMENDATION</h4>
          {rec ? parseRecText(rec) : <p style={{opacity: 0.4}}>Awaiting strategy goals...</p>}
        </div>
        {/* Knowledge graph — full height */}
        <div style={{flex: 1, background: '#000', borderRadius: '25px', overflow: 'hidden', minHeight: '400px'}}>
          {kg ? (
            <Plot 
              data={kg.data}
              layout={{...kg.layout, autosize: false, width: 520, height: 460,
                paper_bgcolor: '#000', plot_bgcolor: '#000',
                margin: {l:0, r:0, b:10, t:40},
                scene: kg.layout.scene ? {...kg.layout.scene, bgcolor: '#000'} : undefined
              }}
              style={{width: '100%'}}
            />
          ) : <div style={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.3}}><GitBranch /></div>}
        </div>
      </div>
    </div>
  );
};

const TraitExplorerStep = ({ traits }) => {
  const [t, setT] = useState({t1: 'Yield_per_plant', t2: 'Height', t3: 'Grain_weight'});
  const [fig, setFig] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/traits?t1=${t.t1}&t2=${t.t2}&t3=${t.t3}`).then(res => setFig(res.data));
  }, [t]);

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <div style={{display: 'flex', gap: '10px', marginBottom: '20px'}}>
        <select value={t.t1} onChange={e => setT({...t, t1:e.target.value})} className="mini-select">{traits.map(tr => <option key={tr} value={tr}>{tr}</option>)}</select>
        <select value={t.t2} onChange={e => setT({...t, t2:e.target.value})} className="mini-select">{traits.map(tr => <option key={tr} value={tr}>{tr}</option>)}</select>
        <select value={t.t3} onChange={e => setT({...t, t3:e.target.value})} className="mini-select">{traits.map(tr => <option key={tr} value={tr}>{tr}</option>)}</select>
      </div>
      <div style={{flex: 1, background: '#000', borderRadius: '25px', overflow: 'hidden'}}>
        {fig && <Plot 
           data={fig.data}
           layout={{...fig.layout, autosize: true, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: {l:0,r:0,b:0,t:0}}}
           useResizeHandler={true}
           style={{width: '100%', height: '100%'}}
        />}
      </div>
    </div>
  );
};

const KnowledgeGraphStep = () => {
  const [fig, setFig] = useState(null);

  useEffect(() => {
    axios.get(`${API_BASE}/api/kg`).then(res => setFig(res.data));
  }, []);

  return (
    <div style={{height: '100%'}}>
      <div style={{height: '100%', background: '#000', borderRadius: '30px', overflow: 'hidden'}}>
        {fig ? <Plot 
           data={fig.data}
           layout={{...fig.layout, autosize: true, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', margin: {l:0,r:0,b:0,t:0}}}
           useResizeHandler={true}
           style={{width: '100%', height: '100%'}}
        /> : <div style={{height:'100%', display:'flex', alignItems:'center', justifyContent:'center'}}><Loader2 className="animate-spin" /></div>}
      </div>
    </div>
  );
};

const MapStep = ({ genotypes }) => {
  const [fig, setFig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [variety, setVariety] = useState('');

  const fetchMap = async (v) => {
    setLoading(true);
    const url = v
      ? `${API_BASE}/api/mapfig?variety=${encodeURIComponent(v)}`
      : `${API_BASE}/api/mapfig`;
    try {
      const res = await axios.get(url);
      setFig(res.data);
    } catch(e) { setFig(null); }
    setLoading(false);
  };

  useEffect(() => { fetchMap(''); }, []);

  const onSelect = (v) => { setVariety(v); fetchMap(v); };

  return (
    <div style={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      {/* Variety dropdown selector */}
      <div style={{display: 'flex', gap: '10px', marginBottom: '10px', alignItems: 'center'}}>
        <select
          className="custom-select"
          style={{flex: 1}}
          value={variety}
          onChange={e => onSelect(e.target.value)}
        >
          <option value="">🗺️ Show All States (Overview)</option>
          {(genotypes || []).map(g => <option key={g} value={g}>{g}</option>)}
        </select>
        {variety && (
          <button className="gold-btn" style={{padding:'10px 18px'}} onClick={() => onSelect('')}>Clear</button>
        )}
      </div>
      {variety && (
        <p style={{color:'rgba(255,255,255,0.55)', fontSize:'0.82rem', marginBottom:'8px', textAlign:'center'}}>
          Hover a pin to see the variety count in that state for <strong style={{color:'#a8ff78'}}>{variety}</strong>
        </p>
      )}

      {/* Map container */}
      <div style={{flex: 1, background: '#000', borderRadius: '20px', overflow: 'hidden', minHeight: '420px'}}>
        {loading && (
          <div style={{height: '420px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '15px'}}>
            <Loader2 className="animate-spin" size={40} color="#a8ff78" />
            <p>Loading India Map...</p>
          </div>
        )}
        {!loading && fig && fig.data && (
          <Plot
            data={fig.data}
            layout={{
              ...fig.layout,
              autosize: true,
              paper_bgcolor: '#000',
              plot_bgcolor: '#000',
              margin: {l: 0, r: 0, t: 50, b: 0},
              font: {color: 'white'},
            }}
            useResizeHandler={true}
            style={{width: '100%', height: '100%'}}
          />
        )}
        {!loading && (!fig || !fig.data) && (
          <div style={{height: '420px', display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: 0.5}}>
            <p>No cultivated states found for this variety</p>
          </div>
        )}
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, unit }) => (
  <div className="metric-card">
    <h3>{label}</h3>
    <p>{parseFloat(value) || value}{unit}</p>
  </div>
);

export default App;
