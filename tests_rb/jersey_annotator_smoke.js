// SPANA engine v2-rebuild — REBUILT 2026-10 — NOT ORIGINAL SOURCE
// Browser smoke test of tools_rb/r2_jersey_annotator.html on a SYNTHETIC pack (driven by test_r2_jersey_annotator_browser.py).
// usage: node jersey_annotator_smoke.js PACK_DIR OUT_JSON
const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1600,height:1000},acceptDownloads:true});
const errs=[];p.on('pageerror',e=>errs.push(String(e)));p.on('console',m=>{if(m.type()==='error')errs.push(m.text())});
await p.goto('file://'+process.argv[2]+'/index.html');
await p.evaluate(()=>localStorage.clear());await p.reload();
await p.fill('#annotator','smoke');await p.dispatchEvent('#annotator','change');
await p.fill('#tDesc','dark kit left');
await p.click('#tTeam >> text=A');                       // blur of #tDesc fires its change during this click
await p.fill('#tNum','7');
await p.click('#tVis >> text=READABLE IN SOME FRAMES');  // regression: the click must survive the #tNum change refresh
const cv=await p.$('#cv');const bb=await cv.boundingBox();
for(let k=0;k<16;k++){
  if(k==2){await p.click('#kVis >> text=OUT OF FRAME');}
  else{await p.click('#kVis >> text=VISIBLE');
    await p.mouse.move(bb.x+bb.width*0.1,bb.y+bb.height*0.2);await p.mouse.down();await p.mouse.move(bb.x+bb.width*0.15,bb.y+bb.height*0.5,{steps:4});await p.mouse.up();
    if(k==1){await p.click('#kNum >> text=UNREADABLE');await p.click('#kWhy >> text=FRONT ON');} else await p.keyboard.press('r');}
  await p.keyboard.press('ArrowRight');}
await p.fill('#pDesc','same kit right');await p.selectOption('#pRel','SAME_TEAM');await p.fill('#pNum','8');await p.click('#bAddP');
await p.click('#strip button >> nth=0');await p.check('#people input[type=checkbox]');
await p.click('#bCheck');const check=await p.textContent('#msgs');
const [d]=await Promise.all([p.waitForEvent('download'),p.click('#btnSave')]);await d.saveAs(process.argv[3]);
console.log(JSON.stringify({check,errs}));await b.close();})();
