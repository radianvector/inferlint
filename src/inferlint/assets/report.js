(function(){
  // One theme button, as on radianvector.com: each click flips light and dark. Only a
  // stored choice changes the theme on load; a data-theme set by whoever hosts the page
  // is left alone, and until a click the page follows the system setting.
  var root=document.documentElement,KEY='inferlint-theme';
  var toggles=document.querySelectorAll('[data-theme-toggle]');
  var mq=window.matchMedia?window.matchMedia('(prefers-color-scheme: dark)'):null;
  function current(){var a=root.getAttribute('data-theme');
    return a==='light'||a==='dark'?a:(mq&&mq.matches?'dark':'light');}
  function label(){var next=current()==='light'?'dark':'light';
    toggles.forEach(function(b){b.setAttribute('aria-label','Switch to '+next+' theme');b.title='Switch to '+next+' theme';});}
  var saved=null;try{saved=localStorage.getItem(KEY);}catch(e){}
  if(saved==='light'||saved==='dark')root.setAttribute('data-theme',saved);
  label();
  if(mq&&mq.addEventListener)mq.addEventListener('change',label);
  toggles.forEach(function(b){b.addEventListener('click',function(){
    var next=current()==='light'?'dark':'light';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem(KEY,next);}catch(e){}
    label();
  });});
  var tt=document.querySelector('.tt');
  function row(cls,val,label){
    var r=document.createElement('div');r.className='row';
    var k=document.createElement('i');k.className='k '+cls;
    var b=document.createElement('b');b.textContent=val;
    var s=document.createElement('span');s.textContent=label;
    r.appendChild(k);r.appendChild(b);r.appendChild(s);return r;
  }
  function place(x,y){
    var w=tt.offsetWidth,h=tt.offsetHeight,vw=window.innerWidth,vh=window.innerHeight;
    var left=x+14,top=y+14;
    if(left+w>vw-8)left=Math.max(8,x-w-14);
    if(top+h>vh-8)top=Math.max(8,y-h-14);
    tt.style.left=left+'px';tt.style.top=top+'px';
  }
  function fmt(v){return v===null||v===undefined?'–':(Math.abs(v)>=100?Math.round(v).toLocaleString():(Math.round(v*10)/10).toLocaleString());}
  document.querySelectorAll('figure[data-chart]').forEach(function(fig){
    var node=fig.querySelector('script.chart-data');if(!node)return;
    var d=JSON.parse(node.textContent);var svg=fig.querySelector('svg');
    var hit=svg.querySelector('.hit'),cross=svg.querySelector('.cross');if(!hit)return;
    var idx=d.x.length-1;
    function nearest(xv){var lo=0,hi=d.x.length-1;while(hi-lo>1){var m=(lo+hi)>>1;if(d.x[m]<xv)lo=m;else hi=m;}
      return (xv-d.x[lo])<(d.x[hi]-xv)?lo:hi;}
    function px(i){return d.px0+(d.x[i]-d.x0)/((d.x1-d.x0)||1)*(d.px1-d.px0);}
    function show(i,cx,cy){
      idx=i;var X=px(i);cross.setAttribute('x1',X);cross.setAttribute('x2',X);cross.setAttribute('visibility','visible');
      tt.textContent='';var h=document.createElement('div');h.className='t-h';
      h.textContent=d.x[i].toFixed(2)+' '+d.xUnit;tt.appendChild(h);
      d.series.forEach(function(s){tt.appendChild(row(s.cls,fmt(s.ys[i]),s.label));});
      var lo=i>0?d.x[i-1]:-Infinity;
      d.events.forEach(function(e){if(e.x>lo&&e.x<=d.x[i])tt.appendChild(row('ev','',e.label));});
      tt.hidden=false;place(cx,cy);
    }
    function hide(){tt.hidden=true;cross.setAttribute('visibility','hidden');}
    hit.addEventListener('pointermove',function(e){
      var r=svg.getBoundingClientRect();var vx=(e.clientX-r.left)*d.w/r.width;
      var xv=d.x0+(vx-d.px0)/(d.px1-d.px0)*(d.x1-d.x0);show(nearest(xv),e.clientX,e.clientY);});
    hit.addEventListener('pointerleave',hide);
    hit.addEventListener('blur',hide);
    hit.addEventListener('focus',function(){var r=hit.getBoundingClientRect();show(idx,r.left+r.width/2,r.top);});
    hit.addEventListener('keydown',function(e){
      var step=e.shiftKey?10:1;
      if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();
        var i=Math.max(0,Math.min(d.x.length-1,idx+(e.key==='ArrowRight'?step:-step)));
        var r=svg.getBoundingClientRect();show(i,r.left+px(i)*r.width/d.w,r.top+20);}
      if(e.key==='Escape')hide();});
  });
  document.querySelectorAll('[data-tip]').forEach(function(el){
    function on(e){tt.textContent=el.getAttribute('data-tip');tt.hidden=false;
      var r=el.getBoundingClientRect();place(e&&e.clientX?e.clientX:r.right,e&&e.clientY?e.clientY:r.top);}
    el.addEventListener('pointerenter',on);el.addEventListener('pointermove',on);el.addEventListener('focus',function(){on(null);});
    el.addEventListener('pointerleave',function(){tt.hidden=true;});el.addEventListener('blur',function(){tt.hidden=true;});
  });
})();
