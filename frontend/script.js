/* TeleOptAI - Frontend Script (API mode) */
var API='http://localhost:5050/api';
var state={running:false,automationMode:false,simSeconds:0,historyLen:60,_prevKpi:null,anomalies:[],aiCooldown:0,pendingParams:null,sendTimer:null,
  params:{bandwidth:{ue1_enodeb:10,enodeb_core:100,core_server:50,ue2_server:20},queue_size:{ue1:100,ue2:100,enodeb:200,core_router:150,server:100},scheduling:{enodeb:'WFQ',core_router:'WFQ'},traffic_load:{ue1:0.60,ue2:0.40}},
  kpi:{throughput:0,latency:0,loss:0,util:0},
  linkSat:{ue1_enodeb:0,enodeb_core:0,core_server:0,ue2_server:0}
};
var ALGO_INFO={
  FIFO:'FIFO: First-In First-Out. Higher latency under load.',
  WFQ:'WFQ: Weighted Fair Queueing. Low jitter, balanced flows.',
  PQ:'PQ: Priority Queueing. Lowest latency for critical traffic.',
  RR:'RR: Round Robin. Fair time slices per flow.'
};
function computeLinkSat(){
  var p=state.params,d1=p.traffic_load.ue1*p.bandwidth.ue1_enodeb,d2=p.traffic_load.ue2*p.bandwidth.ue2_server;
  return{ue1_enodeb:p.traffic_load.ue1,enodeb_core:Math.min(d1/p.bandwidth.enodeb_core,1),core_server:Math.min((d1+d2)/p.bandwidth.core_server,1),ue2_server:p.traffic_load.ue2};
}
function manualControlsLocked(){return state.running&&state.automationMode;}
function presetControlsLocked(){return false;}
function snapshotParams(){
  return{
    bandwidth:Object.assign({},state.params.bandwidth),
    queue_size:Object.assign({},state.params.queue_size),
    scheduling_algorithm:Object.assign({},state.params.scheduling),
    traffic_load:Object.assign({},state.params.traffic_load)
  };
}
function syncControlModeUI(){
  var manualBtn=document.getElementById('mode-manual'),autoBtn=document.getElementById('mode-auto'),note=document.getElementById('modeNote');
  if(manualBtn)manualBtn.classList.toggle('active',!state.automationMode);
  if(autoBtn)autoBtn.classList.toggle('active',state.automationMode);
  if(note)note.textContent=state.automationMode
    ?'Automated allocation is enabled. Use scenario presets to seed the issue, and the orchestrator will adjust resources dynamically while the simulation runs.'
    :'Manual control is enabled. Use the sliders and presets to configure the scenario.';
  document.querySelectorAll('.slider,.algo-pill,.preset-btn').forEach(function(el){
    var lock=el.classList.contains('preset-btn') ? presetControlsLocked() : manualControlsLocked();
    el.disabled=lock;
    el.classList.toggle('disabled-control',lock);
  });
}
async function setControlMode(enabled){
  state.automationMode=!!enabled;
  if(state.automationMode)state.pendingParams=null;
  syncControlModeUI();
  await apiPost('/control-mode',{automation_enabled:state.automationMode});
}
async function apiPost(ep,body){
  try{var r=await fetch(API+ep,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.ok?await r.json():null;}catch(e){return null;}
}
async function apiGet(ep){
  try{var r=await fetch(API+ep);return r.ok?await r.json():null;}catch(e){return null;}
}
function applyBackendParameters(params){
  if(!params)return;
  state.params.bandwidth=Object.assign({},state.params.bandwidth,params.bandwidth||{});
  state.params.queue_size=Object.assign({},state.params.queue_size,params.queue_size||{});
  if(params.scheduling_algorithm){
    state.params.scheduling={
      enodeb:params.scheduling_algorithm.enodeb||state.params.scheduling.enodeb,
      core_router:params.scheduling_algorithm.core_router||state.params.scheduling.core_router
    };
  }

  var bwLabels={ue1_enodeb:'bw-ue1-val',enodeb_core:'bw-enodeb-val',core_server:'bw-core-val',ue2_server:'bw-ue2-val'};
  Object.keys(bwLabels).forEach(function(link){
    var val=state.params.bandwidth[link];
    var inputId=link==='ue1_enodeb'?'bw-ue1':link==='enodeb_core'?'bw-enodeb':link==='core_server'?'bw-core':'bw-ue2';
    var input=document.getElementById(inputId),label=document.getElementById(bwLabels[link]);
    if(input)input.value=val;
    if(label)label.textContent=val+' Mbps';
  });

  var queueLabels={ue1:'q-ue1-val',ue2:'q-ue2-val',enodeb:'q-enodeb-val',core_router:'q-core-val',server:'q-server-val'};
  Object.keys(queueLabels).forEach(function(node){
    var val=state.params.queue_size[node];
    var inputId=node==='ue1'?'q-ue1':node==='ue2'?'q-ue2':node==='enodeb'?'q-enodeb':node==='core_router'?'q-core':'q-server';
    var input=document.getElementById(inputId),label=document.getElementById(queueLabels[node]);
    if(input)input.value=val;
    if(label)label.textContent=val;
  });

  ['enodeb','core_router'].forEach(function(node){
    var algo=state.params.scheduling[node];
    document.querySelectorAll('.algo-pills[data-node="'+node+'"] .algo-pill').forEach(function(p){
      p.classList.toggle('active',p.dataset.algo===algo);
    });
  });
}
function scheduleSendParams(){clearTimeout(state.sendTimer);state.sendTimer=setTimeout(flushParams,400);}
async function flushParams(){
  if(!state.running||!state.pendingParams||manualControlsLocked())return;
  var res=await apiPost('/parameters',{
    bandwidth:state.pendingParams.bandwidth||{},
    queue_size:state.pendingParams.queue_size||{},
    scheduling_algorithm:state.pendingParams.scheduling||{},
    traffic_load:state.pendingParams.traffic_load||{}
  });
  if(res&&res.changes)res.changes.forEach(function(c){addLogEntry(c);});
  state.pendingParams=null;
}
async function pollKPIs(){
  var data=await apiGet('/kpis');
  if(!data||!data.kpi)return;
  var params=await apiGet('/parameters');
  if(params&&params.parameters){
    applyBackendParameters(params.parameters);
    state.linkSat=computeLinkSat();
  }
  var kpi=data.kpi;
  state._prevKpi=Object.assign({},state.kpi);
  state.kpi={throughput:kpi.throughput,latency:kpi.latency,loss:kpi.packet_loss,util:kpi.utilization};
  state.anomalies=(data.anomalies||[]).map(function(a){
    var hi=(a.severity==='critical'||a.severity==='high');
    return{level:hi?'high':a.severity==='medium'?'medium':'low',icon:hi?'[H]':'[M]',
      title:a.type.split('_').map(function(w){return w[0].toUpperCase()+w.slice(1);}).join(' '),
      desc:'Confidence '+(a.confidence*100).toFixed(0)+'% on '+a.nodes.join(', ')};
  });
  state.linkSat=computeLinkSat();
  state.simSeconds++;
  document.getElementById('simTime').textContent=formatSimTime(state.simSeconds);
  updateKPICards();updateSparklines();pushToCharts(formatSimTime(state.simSeconds));
  updateLinkBars();updateAnomalyPanel();updateAIPanel();updateStatusBySeverity();
}

/* === Topology Canvas === */
var TOPO_NODES={
  ue1:{label:'UE1',type:'UE',color:'#6366f1',rx:0.10,ry:0.45},
  enodeb:{label:'eNodeB',type:'eNodeB',color:'#06b6d4',rx:0.35,ry:0.45},
  core_router:{label:'Core',type:'CoreRouter',color:'#f59e0b',rx:0.60,ry:0.45},
  server:{label:'Server',type:'Server',color:'#10b981',rx:0.85,ry:0.45},
  ue2:{label:'UE2',type:'UE',color:'#6366f1',rx:0.85,ry:0.80}
};
var TOPO_LINKS=[
  {from:'ue1',to:'enodeb',key:'ue1_enodeb'},{from:'enodeb',to:'core_router',key:'enodeb_core'},
  {from:'core_router',to:'server',key:'core_server'},{from:'ue2',to:'server',key:'ue2_server'}
];
var topoCanvas,topoCtx,packets=[];
function initTopology(){
  topoCanvas=document.getElementById('topoCanvas');topoCtx=topoCanvas.getContext('2d');
  resizeTopology();window.addEventListener('resize',resizeTopology);spawnPackets();requestAnimationFrame(drawTopology);
}
function resizeTopology(){var r=topoCanvas.parentElement.getBoundingClientRect();topoCanvas.width=r.width-28;topoCanvas.height=220;}
function nodePos(id){var n=TOPO_NODES[id];return{x:n.rx*topoCanvas.width,y:n.ry*topoCanvas.height};}
function spawnPackets(){
  packets=[];
  TOPO_LINKS.forEach(function(lnk){for(var i=0;i<3;i++)packets.push({link:lnk,t:Math.random(),speed:0.005+Math.random()*0.008});});
}
function drawTopology(){
  var c=topoCtx,W=topoCanvas.width,H=topoCanvas.height;
  c.clearRect(0,0,W,H);
  c.strokeStyle='rgba(99,102,241,0.04)';c.lineWidth=1;
  for(var x=0;x<W;x+=40){c.beginPath();c.moveTo(x,0);c.lineTo(x,H);c.stroke();}
  for(var y=0;y<H;y+=40){c.beginPath();c.moveTo(0,y);c.lineTo(W,y);c.stroke();}
  TOPO_LINKS.forEach(function(lnk){
    var a=nodePos(lnk.from),b=nodePos(lnk.to),sat=state.running?(state.linkSat[lnk.key]||0):0;
    var hue=sat<0.7?220:sat<0.9?40:0;
    c.beginPath();c.moveTo(a.x,a.y);c.lineTo(b.x,b.y);
    c.strokeStyle='hsla('+hue+',80%,60%,'+(0.3+sat*0.5)+')';c.lineWidth=1.5+sat*2.5;c.setLineDash([]);c.stroke();
    var mx=(a.x+b.x)/2,my=(a.y+b.y)/2-10;
    c.fillStyle='rgba(148,163,184,0.7)';c.font='9px monospace';c.textAlign='center';
    c.fillText(state.params.bandwidth[lnk.key]+' Mbps',mx,my);
  });
  if(state.running){
    packets.forEach(function(pkt){
      pkt.t=(pkt.t+pkt.speed)%1;
      var a=nodePos(pkt.link.from),b=nodePos(pkt.link.to);
      var px=a.x+(b.x-a.x)*pkt.t,py=a.y+(b.y-a.y)*pkt.t;
      var sat=state.linkSat[pkt.link.key]||0,hue=sat<0.7?200:sat<0.9?40:0;
      c.beginPath();c.arc(px,py,2.5,0,Math.PI*2);
      c.fillStyle='hsla('+hue+',90%,70%,0.9)';c.shadowColor='hsla('+hue+',90%,70%,0.8)';c.shadowBlur=6;c.fill();c.shadowBlur=0;
    });
  }
  Object.keys(TOPO_NODES).forEach(function(id){
    var n=TOPO_NODES[id],p=nodePos(id),r=22,x=p.x,y=p.y;
    if(state.running){var g=c.createRadialGradient(x,y,r,x,y,r+12);g.addColorStop(0,n.color+'44');g.addColorStop(1,n.color+'00');c.beginPath();c.arc(x,y,r+12,0,Math.PI*2);c.fillStyle=g;c.fill();}
    c.beginPath();c.arc(x,y,r,0,Math.PI*2);
    var f=c.createRadialGradient(x-6,y-6,2,x,y,r);f.addColorStop(0,n.color+'cc');f.addColorStop(1,n.color+'55');
    c.fillStyle=f;c.strokeStyle=n.color;c.lineWidth=1.5;c.fill();c.stroke();
    c.fillStyle='#f1f5f9';c.font='bold 10px sans-serif';c.textAlign='center';c.fillText(n.label,x,y+4);
    c.fillStyle=n.color+'aa';c.font='8px sans-serif';c.fillText(n.type,x,y+r+12);
  });
  requestAnimationFrame(drawTopology);
}

/* === Sparklines === */
var sparkData={throughput:[],latency:[],loss:[],util:[]};
var sparkCharts={};
function initSparklines(){
  [{id:'spark-throughput',key:'throughput',color:'#818cf8'},{id:'spark-latency',key:'latency',color:'#f59e0b'},
   {id:'spark-loss',key:'loss',color:'#f43f5e'},{id:'spark-util',key:'util',color:'#06b6d4'}].forEach(function(d){
    var canvas=document.getElementById(d.id);
    sparkCharts[d.key]=new Chart(canvas,{type:'line',
      data:{labels:Array(20).fill(''),datasets:[{data:Array(20).fill(null),borderColor:d.color,borderWidth:1.5,fill:true,backgroundColor:d.color+'22',pointRadius:0,tension:0.4}]},
      options:{animation:false,responsive:false,plugins:{legend:{display:false},tooltip:{enabled:false}},scales:{x:{display:false},y:{display:false}}}});
  });
}
function updateSparklines(){
  ['throughput','latency','loss','util'].forEach(function(k){
    var d=sparkData[k];d.push(state.kpi[k]);if(d.length>20)d.shift();
    var ch=sparkCharts[k];ch.data.datasets[0].data=d.slice();ch.update('none');
  });
}

/* === Main Charts === */
var chartTx,chartLat;
function initCharts(){
  var baseOpts={animation:{duration:300},responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#94a3b8',font:{size:11},boxWidth:12}},tooltip:{mode:'index',intersect:false,backgroundColor:'#111827',titleColor:'#f1f5f9',bodyColor:'#94a3b8'}},
    scales:{x:{ticks:{color:'#475569',font:{size:9},maxTicksLimit:10},grid:{color:'rgba(255,255,255,0.04)'}},
            y:{ticks:{color:'#475569',font:{size:9}},grid:{color:'rgba(255,255,255,0.04)'}}}};
  chartTx=new Chart(document.getElementById('chartThroughput'),{type:'line',
    data:{labels:[],datasets:[
      {label:'Throughput (Mbps)',data:[],borderColor:'#818cf8',backgroundColor:'#818cf822',borderWidth:2,fill:true,pointRadius:0,tension:0.4,yAxisID:'y'},
      {label:'Utilization (%)',data:[],borderColor:'#06b6d4',backgroundColor:'#06b6d411',borderWidth:2,fill:false,pointRadius:0,tension:0.4,yAxisID:'y1'}]},
    options:Object.assign({},baseOpts,{scales:{x:baseOpts.scales.x,y:Object.assign({},baseOpts.scales.y,{position:'left'}),y1:Object.assign({},baseOpts.scales.y,{position:'right',grid:{drawOnChartArea:false}})}})});
  chartLat=new Chart(document.getElementById('chartLatency'),{type:'line',
    data:{labels:[],datasets:[
      {label:'Latency (ms)',data:[],borderColor:'#f59e0b',backgroundColor:'#f59e0b22',borderWidth:2,fill:true,pointRadius:0,tension:0.4,yAxisID:'y'},
      {label:'Packet Loss (%)',data:[],borderColor:'#f43f5e',backgroundColor:'#f43f5e11',borderWidth:2,fill:false,pointRadius:0,tension:0.4,yAxisID:'y1'}]},
    options:Object.assign({},baseOpts,{scales:{x:baseOpts.scales.x,y:Object.assign({},baseOpts.scales.y,{position:'left'}),y1:Object.assign({},baseOpts.scales.y,{position:'right',grid:{drawOnChartArea:false}})}})});
}
function pushToCharts(label){
  function push(chart,vals){
    chart.data.labels.push(label);vals.forEach(function(v,i){chart.data.datasets[i].data.push(v);});
    if(chart.data.labels.length>state.historyLen){chart.data.labels.shift();chart.data.datasets.forEach(function(d){d.data.shift();});}
    chart.update('none');
  }
  push(chartTx,[+state.kpi.throughput.toFixed(2),+state.kpi.util.toFixed(1)]);
  push(chartLat,[+state.kpi.latency.toFixed(1),+state.kpi.loss.toFixed(2)]);
}

/* === DOM Updaters === */
function formatSimTime(s){var h=String(Math.floor(s/3600)).padStart(2,'0'),m=String(Math.floor((s%3600)/60)).padStart(2,'0'),sec=String(s%60).padStart(2,'0');return h+':'+m+':'+sec;}
function lerp(a,b,t){return a+(b-a)*t;}
function updateKPICards(){
  var k=state.kpi,p=state._prevKpi||{};
  function setVal(id,v,d){document.getElementById(id).textContent=v.toFixed(d);}
  function setTrend(id,cur,old,goodUp){
    var el=document.getElementById(id),up=old&&cur>old*1.02,dn=old&&cur<old*0.98;
    el.textContent=up?'up':dn?'dn':'ok';
    el.className='kpi-trend '+(up?(goodUp?'up':'down'):dn?(goodUp?'down':'up'):'');
  }
  setVal('val-throughput',k.throughput,1);setVal('val-latency',k.latency,0);
  setVal('val-loss',k.loss,2);setVal('val-util',k.util,1);
  setTrend('trend-throughput',k.throughput,p.throughput,true);
  setTrend('trend-latency',k.latency,p.latency,false);
  setTrend('trend-loss',k.loss,p.loss,false);
  setTrend('trend-util',k.util,p.util,true);
  document.getElementById('kpi-loss').classList.toggle('alert-high',k.loss>5);
  document.getElementById('kpi-loss').classList.toggle('alert-warn',k.loss>2&&k.loss<=5);
  document.getElementById('kpi-latency').classList.toggle('alert-high',k.latency>100);
  document.getElementById('kpi-util').classList.toggle('alert-warn',k.util>85);
}
function updateLinkBars(){
  [{key:'ue1_enodeb',f:'lb-ue1',p:'lbp-ue1'},{key:'enodeb_core',f:'lb-enodeb',p:'lbp-enodeb'},
   {key:'core_server',f:'lb-core',p:'lbp-core'},{key:'ue2_server',f:'lb-ue2',p:'lbp-ue2'}].forEach(function(l){
    var pct=Math.round((state.linkSat[l.key]||0)*100);
    var fill=document.getElementById(l.f),pctEl=document.getElementById(l.p);
    fill.style.width=pct+'%';
    fill.className='link-bar-fill'+(pct>90?' critical':pct>70?' warning':'');
    pctEl.textContent=pct+'%';
  });
}
function updateAnomalyPanel(){
  var list=document.getElementById('alertsList'),badge=document.getElementById('anomalyCount'),anoms=state.anomalies;
  badge.textContent=anoms.length;badge.className='badge'+(anoms.length===0?' zero':'');
  if(anoms.length===0){list.innerHTML='<div class="no-alerts">No anomalies detected</div>';return;}
  var now=new Date().toLocaleTimeString();
  list.innerHTML=anoms.map(function(a){
    return '<div class="alert-item '+a.level+'"><span class="alert-icon">'+a.icon+'</span><div class="alert-body"><div class="alert-title">'+a.title+'</div><div class="alert-desc">'+a.desc+'</div></div><span class="alert-time">'+now+'</span></div>';
  }).join('');
}
function updateAIPanel(){
  if(state.aiCooldown-->0)return;state.aiCooldown=5;
  var k=state.kpi,p=state.params,insights=[],colors={INCREASE_CAPACITY:'#06b6d4',OPTIMIZE_QUEUE:'#f59e0b',SCHEDULE_CHANGE:'#8b5cf6',REDUCE_CAPACITY:'#10b981',OK:'#6366f1'};
  if(k.util>80){var bot=Object.entries(p.bandwidth).sort(function(a,b){return a[1]-b[1];})[0];insights.push({a:'INCREASE_CAPACITY',t:'Utilization '+k.util.toFixed(0)+'%. Recommend increasing '+bot[0].replace(/_/g,'<>>')+' bandwidth from '+bot[1]+' Mbps.'});}
  if(k.loss>3)insights.push({a:'OPTIMIZE_QUEUE',t:'Packet loss '+k.loss.toFixed(1)+'% > 3%. Increase queue size or switch to PQ scheduling.'});
  if(k.latency>80&&p.scheduling.enodeb!=='PQ')insights.push({a:'SCHEDULE_CHANGE',t:'Latency '+k.latency.toFixed(0)+'ms high. Switch eNodeB to PQ for delay-sensitive flows.'});
  if(insights.length===0)insights.push({a:'OK',t:'Network nominal. Throughput '+k.throughput.toFixed(1)+' Mbps, latency '+k.latency.toFixed(0)+'ms, loss '+k.loss.toFixed(2)+'%.'});
  document.getElementById('aiPanel').innerHTML=insights.map(function(ins){
    return '<div class="ai-insight"><div class="ai-insight-action" style="color:'+(colors[ins.a]||'#6366f1')+'">'+ins.a.replace(/_/g,' ')+'</div><div class="ai-insight-text">'+ins.t+'</div></div>';
  }).join('');
}
function updateStatusBySeverity(){
  var k=state.kpi;
  if(k.loss>5||k.latency>100)updateStatusBar('Warning -- Anomalies Detected','warning');
  else updateStatusBar('Simulation Running','running');
}
function addLogEntry(html){
  var c=document.getElementById('logContainer'),now=new Date().toLocaleTimeString();
  var el=document.createElement('div');el.className='log-entry';
  el.innerHTML='<span class="log-time">'+now+'</span><span class="log-text">'+html+'</span>';
  c.prepend(el);c.querySelectorAll('.log-empty').forEach(function(e){e.remove();});
  var ents=c.querySelectorAll('.log-entry');if(ents.length>30)ents[ents.length-1].remove();
}
function clearLog(){document.getElementById('logContainer').innerHTML='<div class="log-empty">No changes yet.</div>';}
function updateStatusBar(text,st){
  document.getElementById('statusText').textContent=text;
  document.getElementById('statusDot').className='status-dot '+(st||'');
}

/* === Simulation Control === */
document.getElementById('startStopBtn').addEventListener('click',async function(){
  var btn=document.getElementById('startStopBtn');
  if(!state.running){
    updateStatusBar('Connecting to backend...','');
    var res=await apiPost('/start',{automation_enabled:state.automationMode,initial_parameters:snapshotParams()});
    if(!res){updateStatusBar('Backend unreachable -- start python backend/api.py','error');return;}
    state.running=true;state.simSeconds=0;
    var startParams=await apiGet('/parameters');
    if(startParams&&startParams.parameters){applyBackendParameters(startParams.parameters);state.linkSat=computeLinkSat();}
    state.pollTimer=setInterval(pollKPIs,1000);
    btn.textContent='Stop Simulation';btn.classList.add('running');
    updateStatusBar('Simulation Running','running');spawnPackets();
    syncControlModeUI();
    addLogEntry('Simulation <strong>started</strong>');
  } else {
    clearInterval(state.pollTimer);state.running=false;
    await apiPost('/stop',{});
    btn.textContent='Start Simulation';btn.classList.remove('running');
    updateStatusBar('Simulation Stopped','');
    syncControlModeUI();
    addLogEntry('Simulation <strong>stopped</strong>');
  }
});

/* === Parameter Controls === */
function updateBandwidth(link,value){
  if(manualControlsLocked())return;
  var val=parseFloat(value),old=state.params.bandwidth[link];
  state.params.bandwidth[link]=val;
  var labels={ue1_enodeb:'bw-ue1-val',enodeb_core:'bw-enodeb-val',core_server:'bw-core-val',ue2_server:'bw-ue2-val'};
  document.getElementById(labels[link]).textContent=val+' Mbps';
  if(!state.pendingParams)state.pendingParams={};
  if(!state.pendingParams.bandwidth)state.pendingParams.bandwidth={};
  state.pendingParams.bandwidth[link]=val;
  scheduleSendParams();clearPresetActive();
}
function updateQueue(node,value){
  if(manualControlsLocked())return;
  var val=parseInt(value),old=state.params.queue_size[node];
  state.params.queue_size[node]=val;
  var labels={ue1:'q-ue1-val',enodeb:'q-enodeb-val',core_router:'q-core-val',server:'q-server-val'};
  if(labels[node])document.getElementById(labels[node]).textContent=val;
  if(!state.pendingParams)state.pendingParams={};
  if(!state.pendingParams.queue_size)state.pendingParams.queue_size={};
  state.pendingParams.queue_size[node]=val;
  scheduleSendParams();clearPresetActive();
}
function updateTrafficLoad(ue,value){
  if(manualControlsLocked())return;
  var val=parseInt(value);state.params.traffic_load[ue]=val/100;
  var labels={ue1:'load-ue1-val',ue2:'load-ue2-val'};
  document.getElementById(labels[ue]).textContent=val+'%';
  if(!state.pendingParams)state.pendingParams={};
  if(!state.pendingParams.traffic_load)state.pendingParams.traffic_load={};
  state.pendingParams.traffic_load[ue]=val/100;
  scheduleSendParams();
  addLogEntry('Traffic <strong>'+ue.toUpperCase()+'</strong>: <span class="log-new">'+val+'% load</span>');
  clearPresetActive();
}
function updateScheduling(node,algo,btn){
  if(manualControlsLocked())return;
  var old=state.params.scheduling[node];state.params.scheduling[node]=algo;
  btn.closest('.algo-pills').querySelectorAll('.algo-pill').forEach(function(p){p.classList.remove('active');});
  btn.classList.add('active');
  document.getElementById('algoInfo').textContent=ALGO_INFO[algo]||algo;
  if(!state.pendingParams)state.pendingParams={};
  if(!state.pendingParams.scheduling)state.pendingParams.scheduling={};
  state.pendingParams.scheduling[node]=algo;
  scheduleSendParams();clearPresetActive();
}
function clearPresetActive(){document.querySelectorAll('.preset-btn').forEach(function(b){b.classList.remove('active');});}

/* === Presets === */
var PRESETS={
  normal:  {bandwidth:{ue1_enodeb:10,enodeb_core:100,core_server:50,ue2_server:20},queue_size:{ue1:100,ue2:100,enodeb:200,core_router:150,server:100},scheduling:{enodeb:'WFQ',core_router:'WFQ'},traffic_load:{ue1:0.60,ue2:0.40}},
  congestion:{bandwidth:{ue1_enodeb:3,enodeb_core:20,core_server:10,ue2_server:5},queue_size:{ue1:50,ue2:50,enodeb:60,core_router:60,server:50},scheduling:{enodeb:'FIFO',core_router:'FIFO'},traffic_load:{ue1:0.95,ue2:0.90}},
  optimized:{bandwidth:{ue1_enodeb:80,enodeb_core:500,core_server:300,ue2_server:80},queue_size:{ue1:500,ue2:500,enodeb:1000,core_router:800,server:500},scheduling:{enodeb:'PQ',core_router:'PQ'},traffic_load:{ue1:0.50,ue2:0.35}},
  failure:  {bandwidth:{ue1_enodeb:1,enodeb_core:100,core_server:50,ue2_server:20},queue_size:{ue1:30,ue2:100,enodeb:200,core_router:150,server:100},scheduling:{enodeb:'FIFO',core_router:'WFQ'},traffic_load:{ue1:0.95,ue2:0.40}}
};
function applyPreset(name){
  var pr=PRESETS[name];if(!pr)return;
  Object.assign(state.params.bandwidth,pr.bandwidth);Object.assign(state.params.queue_size,pr.queue_size);
  Object.assign(state.params.scheduling,pr.scheduling);Object.assign(state.params.traffic_load,pr.traffic_load);
  document.getElementById('bw-ue1').value=pr.bandwidth.ue1_enodeb;document.getElementById('bw-enodeb').value=pr.bandwidth.enodeb_core;
  document.getElementById('bw-core').value=pr.bandwidth.core_server;document.getElementById('bw-ue2').value=pr.bandwidth.ue2_server;
  document.getElementById('bw-ue1-val').textContent=pr.bandwidth.ue1_enodeb+' Mbps';
  document.getElementById('bw-enodeb-val').textContent=pr.bandwidth.enodeb_core+' Mbps';
  document.getElementById('bw-core-val').textContent=pr.bandwidth.core_server+' Mbps';
  document.getElementById('bw-ue2-val').textContent=pr.bandwidth.ue2_server+' Mbps';
  document.getElementById('q-ue1').value=pr.queue_size.ue1;document.getElementById('q-enodeb').value=pr.queue_size.enodeb;
  document.getElementById('q-core').value=pr.queue_size.core_router;document.getElementById('q-server').value=pr.queue_size.server;
  document.getElementById('q-ue1-val').textContent=pr.queue_size.ue1;document.getElementById('q-enodeb-val').textContent=pr.queue_size.enodeb;
  document.getElementById('q-core-val').textContent=pr.queue_size.core_router;document.getElementById('q-server-val').textContent=pr.queue_size.server;
  document.getElementById('load-ue1').value=Math.round(pr.traffic_load.ue1*100);document.getElementById('load-ue2').value=Math.round(pr.traffic_load.ue2*100);
  document.getElementById('load-ue1-val').textContent=Math.round(pr.traffic_load.ue1*100)+'%';
  document.getElementById('load-ue2-val').textContent=Math.round(pr.traffic_load.ue2*100)+'%';
  ['enodeb','core_router'].forEach(function(node){
    var algo=pr.scheduling[node];
    document.querySelectorAll('.algo-pills[data-node="'+node+'"] .algo-pill').forEach(function(p){p.classList.toggle('active',p.dataset.algo===algo);});
  });
  document.getElementById('algoInfo').textContent=ALGO_INFO[pr.scheduling.enodeb]||'';
  document.querySelectorAll('.preset-btn').forEach(function(b){b.classList.remove('active');});
  document.getElementById('preset-'+name).classList.add('active');
  if(state.running){
    if(state.automationMode){
      apiPost('/parameters',Object.assign({scenario_seed:true},snapshotParams())).then(function(res){
        if(res&&res.changes)res.changes.forEach(function(c){addLogEntry(c);});
      });
    } else {
      state.pendingParams=snapshotParams();
      scheduleSendParams();
    }
  }
  addLogEntry('Preset: <strong>'+name.toUpperCase()+'</strong>'+(state.automationMode ? ' <span class="log-new">(automation seeding)</span>' : ''));
}

/* === Init === */
window.addEventListener('DOMContentLoaded',function(){
  initTopology();initSparklines();initCharts();
  apiGet('/control-mode').then(function(res){
    if(res&&typeof res.automation_enabled==='boolean')state.automationMode=res.automation_enabled;
    syncControlModeUI();
  });
  updateStatusBar('Ready -- Start simulation to begin','');
  updateKPICards();updateLinkBars();
});

