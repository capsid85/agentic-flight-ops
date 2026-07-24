import { useState, useEffect } from 'react';
import axios from 'axios';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
import Map from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { 
  AlertTriangle, ShieldCheck, 
  Database, Compass, Play
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// Free dark basemap from Carto
const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';

const AIRPORT_NAMES: Record<string, string> = {
  "ATL": "Atlanta", "ORD": "Chicago", "DFW": "Dallas", "DEN": "Denver",
  "JFK": "New York", "LAX": "Los Angeles", "SFO": "San Francisco",
  "SEA": "Seattle", "MIA": "Miami", "BOS": "Boston", "CLT": "Charlotte",
  "PHX": "Phoenix", "IAH": "Houston", "MCO": "Orlando", "EWR": "Newark",
  "MSP": "Minneapolis", "DTW": "Detroit", "PHL": "Philadelphia",
  "LGA": "New York", "BWI": "Baltimore", "SLC": "Salt Lake City",
  "SAN": "San Diego", "IAD": "Washington DC", "DCA": "Washington DC",
  "MDW": "Chicago", "TPA": "Tampa", "PDX": "Portland", "HNL": "Honolulu",
  "BNA": "Nashville", "AUS": "Austin", "DAL": "Dallas", "STL": "St. Louis",
  "MSY": "New Orleans", "SMF": "Sacramento", "SJC": "San Jose",
  "SNA": "Orange County", "RDU": "Raleigh", "CLE": "Cleveland",
  "IND": "Indianapolis", "PIT": "Pittsburgh", "SAT": "San Antonio",
  "CVG": "Cincinnati", "CMH": "Columbus", "RSW": "Fort Myers",
  "PBI": "West Palm Beach", "JAX": "Jacksonville", "ANC": "Anchorage",
  "LHR": "London", "CDG": "Paris", "FRA": "Frankfurt", "HND": "Tokyo",
  "DXB": "Dubai", "SYD": "Sydney", "YYZ": "Toronto", "MEX": "Mexico City",
  "HOU": "Houston"
};

const getAirportName = (code: string) => AIRPORT_NAMES[code] || code;

export default function App() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [query, setQuery] = useState("");
  const [liveFlights, setLiveFlights] = useState<any[]>([]);

  const [viewState, setViewState] = useState({
    longitude: -95.7129,
    latitude: 37.0902,
    zoom: 4,
    pitch: 45
  });

  useEffect(() => {
    if (data?.flight) {
      setViewState({
        longitude: data.flight.longitude || -95.7129,
        latitude: data.flight.latitude || 37.0902,
        zoom: 4,
        pitch: 45
      });
    }
  }, [data]);



  // Fetch live background flights like FlightRadar24
  useEffect(() => {
    let interval: any;
    const fetchLiveFlights = async () => {
      try {
        const res = await axios.get("http://localhost:8000/live-flights");
        if (res.data) {
          if (res.data.flights) {
            setLiveFlights(res.data.flights);
          } else if (res.data.states) {
            const flights = res.data.states.slice(0, 800).map((s: any) => ({
              id: s[1],
              lon: s[5],
              lat: s[6],
              is_landing: false
            })).filter((f: any) => f.lon && f.lat);
            setLiveFlights(flights);
          }
        }
      } catch (err) {
        console.error("Failed to fetch live background flights", err);
      }
    };
    
    fetchLiveFlights();
    interval = setInterval(fetchLiveFlights, 10000);
    return () => clearInterval(interval);
  }, []);

  const runReplay = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/analyze', {
        query: query
      });
      setData(response.data);
    } catch (error: any) {
      console.warn("Replay engine warning:", error);
    } finally {
      setLoading(false);
    }
  };

  const getMapData = () => {
    if (!data) return [];
    return [
      {
        sourcePosition: [data.flight.origin_longitude || data.flight.longitude || -95.7129, data.flight.origin_latitude || data.flight.latitude || 37.0902],
        targetPosition: [data.flight.dest_longitude || -95.7129, data.flight.dest_latitude || 37.0902],
        currentPosition: [data.flight.longitude || -95.7129, data.flight.latitude || 37.0902],
        color: data.risk.predicted_delay_minutes > 15 ? [244, 63, 94] : [16, 185, 129], // rose-500 : emerald-500
        radius: 15000 // 15km
      }
    ];
  };

  const layers = [
    new ScatterplotLayer({
      id: 'background-flights',
      data: liveFlights,
      pickable: true,
      opacity: 0.8,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 2,
      radiusMaxPixels: 6,
      getPosition: (d: any) => [d.lon, d.lat],
      getFillColor: (d: any) => (d.is_landing ? [245, 158, 11] : [148, 163, 184]), // amber-500 for landing, slate-400 for cruise
    }),
    new ArcLayer({
      id: 'flight-arc',
      data: getMapData(),
      getSourcePosition: (d: any) => d.sourcePosition,
      getTargetPosition: (d: any) => d.targetPosition,
      getSourceColor: (d: any) => d.color,
      getTargetColor: () => [255, 255, 255],
      getWidth: 4,
      opacity: 0.8,
    }),
    new ScatterplotLayer({
      id: 'flight-origin-dot',
      data: getMapData(),
      pickable: true,
      opacity: 0.9,
      stroked: true,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 6,
      radiusMaxPixels: 15,
      lineWidthMinPixels: 2,
      getPosition: (d: any) => d.sourcePosition,
      getFillColor: (d: any) => d.color,
      getLineColor: () => [255, 255, 255, 255],
      getRadius: (d: any) => d.radius,
    }),
    new ScatterplotLayer({
      id: 'flight-dest-dot',
      data: getMapData(),
      pickable: true,
      opacity: 0.9,
      stroked: true,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 6,
      radiusMaxPixels: 15,
      lineWidthMinPixels: 2,
      getPosition: (d: any) => d.targetPosition,
      getFillColor: () => [255, 255, 255, 255],
      getLineColor: () => [255, 255, 255, 255],
      getRadius: (d: any) => d.radius,
    }),
    new ScatterplotLayer({
      id: 'flight-current-dot',
      data: getMapData(),
      pickable: true,
      opacity: 1,
      stroked: true,
      filled: true,
      radiusScale: 1,
      radiusMinPixels: 4,
      radiusMaxPixels: 10,
      lineWidthMinPixels: 2,
      getPosition: (d: any) => d.currentPosition,
      getFillColor: () => [234, 179, 8, 255], // yellow-500 for the plane
      getLineColor: () => [0, 0, 0, 255],
      getRadius: (d: any) => d.radius * 0.7,
    })
  ];

  return (
    <div className="relative min-h-screen bg-aviation-dark text-slate-200 font-sans overflow-hidden">
      
      {/* BACKGROUND MAP */}
      <div className="absolute inset-0 z-0 opacity-80">
        <DeckGL
          viewState={viewState}
          onViewStateChange={({viewState}) => setViewState(viewState)}
          controller={true}
          layers={layers}
        >
          <Map mapLib={maplibregl as any} mapStyle={MAP_STYLE} />
        </DeckGL>
        {/* Vignette overlay for blending */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-transparent via-aviation-dark/50 to-aviation-dark shadow-inner"></div>
      </div>

      {/* FLOATING UI LAYER */}
      <div className="relative z-10 flex flex-col h-screen pointer-events-none p-6">
        
        {/* TOP NAVBAR */}
        <header className="glass-panel w-full max-w-7xl mx-auto flex justify-between items-center px-6 py-4 pointer-events-auto mb-6">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-indigo-500 to-cyan-400 p-2.5 rounded-xl shadow-[0_0_15px_rgba(99,102,241,0.5)]">
              <Compass className="text-white w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h1 className="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 tracking-tight">
                FLIGHT DASHBOARD
              </h1>
              <div className="flex items-center gap-3 text-[10px] font-mono tracking-widest uppercase">
                <span className="text-cyan-400">Agentic Operations Platform</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <input 
              type="text" placeholder="Flight e.g. DL135" value={query} onChange={(e) => setQuery(e.target.value.toUpperCase())}
              className="bg-slate-900/80 border border-cyan-500/50 text-white px-4 py-2 rounded-xl text-sm font-mono placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 w-44 shadow-inner"
            />
            
            <button 
              onClick={runReplay} 
              disabled={loading || query.trim() === ""} 
              className={`font-bold py-2 px-6 rounded-xl text-sm transition-all flex items-center gap-2 disabled:opacity-50 text-white shadow-lg bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400`}
            >
              {loading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> : <Play className="w-4 h-4" fill="currentColor" />}
              {loading ? "EVALUATING..." : "ANALYZE LIVE FLIGHT"}
            </button>
          </div>
        </header>

        {/* MAIN CONTENT AREA */}
        <div className="flex-1 flex gap-6 max-w-[1600px] w-full mx-auto pb-6">
          
          {/* LEFT SIDE: KPIs & Alerts */}
          <div className="w-[350px] flex flex-col gap-6 pointer-events-auto">
            
            {/* Mission Critical KPIs */}
            <div className="glass-panel p-5 flex flex-col gap-4">
              <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700/50 pb-2 mb-1">Flight Telemetry</h2>
              
              <div className="glass-card">
                <div className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Callsign</div>
                <div className="text-3xl font-extrabold text-white mt-1 tracking-tight">{data?.flight?.flight_id || 'STANDBY'}</div>
                <div className="text-cyan-400 text-xs mt-1 font-mono font-bold flex justify-between items-center">
                  <span>{data ? `${getAirportName(data?.flight?.origin)} ✈ ${getAirportName(data?.flight?.destination)}` : 'Awaiting initialization...'}</span>
                  {data?.flight?.status && <span className="text-[9px] text-amber-400 bg-slate-900/90 px-1.5 py-0.5 rounded border border-amber-500/30">{data.flight.status.split('(')[0]}</span>}
                </div>
              </div>

              <div className="flex gap-2">
                <div className="glass-card flex-1 border-indigo-500/20 p-2.5 flex flex-col justify-between">
                  <div className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Delay Predict</div>
                  <div className="text-xl font-bold text-indigo-400 mt-1 data-glow flex items-baseline gap-1">
                    {data ? (
                      <>
                        <span>{data?.risk?.predicted_delay_minutes || 0}m</span>
                        <span className="text-sm font-mono text-slate-400 font-semibold ml-1">
                          {data?.risk?.confidence_score >= 0.90 ? '±5m' : '±15m'}
                        </span>
                      </>
                    ) : '--'}
                  </div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">
                    {data ? (data?.risk?.confidence_score >= 0.90 ? 'Radar Ground Proximity' : 'ML Variance Estimate') : 'Awaiting Data'}
                  </div>
                </div>
                <div className="glass-card flex-1 border-rose-500/20 p-2.5 flex flex-col justify-between">
                  <div className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Weather Risk</div>
                  <div className={`text-lg font-bold mt-1 ${data?.weather?.dest_risk === 'High' ? 'text-rose-500 data-glow' : 'text-emerald-400'}`}>
                    {data ? (data?.weather?.dest_risk?.toUpperCase() || 'UNK') : '--'}
                  </div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">
                    {data ? `Origin: ${data?.weather?.origin_risk || 'Low'}` : 'Awaiting Data'}
                  </div>
                </div>
                <div className="glass-card flex-1 border-cyan-500/20 p-2.5 flex flex-col justify-between">
                  <div className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Confidence</div>
                  <div className="text-xl font-bold text-cyan-400 mt-1 data-glow">
                    {data ? `${Math.round((data?.risk?.confidence_score || 0) * 100)}%` : '--'}
                  </div>
                  <div className="text-[8px] text-slate-500 font-mono mt-0.5">
                    {data ? (data?.risk?.confidence_score >= 0.90 ? 'High Certainty' : 'Standard Baseline') : 'Awaiting Data'}
                  </div>
                </div>
              </div>
            </div>

            {/* NOTAM / ATC Alert */}
            {data?.notam?.active_notam && (
              <div className="glass-panel p-4 flex flex-col gap-2 border-rose-500/50 bg-rose-950/20 relative overflow-hidden animate-pulse">
                <div className="absolute top-0 right-0 p-2 opacity-10 text-rose-500"><AlertTriangle className="w-12 h-12" /></div>
                <h2 className="text-xs font-bold text-rose-400 uppercase tracking-widest flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4" /> ACTIVE NOTAM: {data.notam.notam_type}
                </h2>
                <div className="text-xs font-mono text-rose-300 leading-relaxed z-10">
                  {data.notam.message}
                </div>
                <div className="bg-rose-900/50 text-rose-200 text-[10px] font-bold px-2 py-1 rounded inline-block w-fit mt-1 border border-rose-500/30">
                  ATC GROUND DELAY PENALTY: +{data.notam.delay_penalty_minutes} MINS
                </div>
              </div>
            )}



          </div>

          {/* MIDDLE: Empty Space for Map interaction */}
          <div className="flex-1 pointer-events-none">
             {/* Map takes this space */}
          </div>

          {/* RIGHT SIDE: Agent Intelligence */}
          <div className="w-[420px] flex flex-col gap-6 pointer-events-auto">
            
            {/* Actionable Directive */}
            <div className="glass-panel p-0 overflow-hidden flex flex-col border-indigo-500/30">
              <div className="bg-gradient-to-r from-indigo-900/80 to-slate-900 p-4 border-b border-indigo-500/30 flex items-center justify-between">
                <h2 className="text-xs font-bold text-indigo-300 uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4" /> Supervisor Directive
                </h2>
                {data && <div className="w-2 h-2 bg-indigo-400 rounded-full animate-ping"></div>}
              </div>
              
              <div className="p-5 flex-1 overflow-y-auto">
                {data ? (
                  <>
                    <div className={`text-xl font-bold mb-4 ${data?.recommendation?.action?.includes('Delay') ? 'text-rose-400' : 'text-emerald-400'}`}>
                      {data?.recommendation?.action}
                    </div>
                    <div className="text-sm text-slate-300 leading-relaxed font-sans mb-4 prose prose-invert max-w-none prose-sm">
                      <ReactMarkdown>
                        {typeof data?.recommendation?.reasoning === 'string' ? data.recommendation.reasoning.replace(/\\n/g, '\n') : (data?.recommendation?.reasoning || '')}
                      </ReactMarkdown>
                    </div>
                    
                    <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-700/50">
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-1">
                        <Database className="w-3 h-3" /> RAG Citation
                      </div>
                      <div className="text-xs font-mono text-slate-400 italic">
                        "{data?.root_cause?.faa_citations?.[0] || 'No mandate cited.'}"
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-500 text-xs font-mono">
                    Awaiting Supervisor Synthesis...
                  </div>
                )}
              </div>
            </div>



          </div>

        </div>
      </div>
    </div>
  );
}
