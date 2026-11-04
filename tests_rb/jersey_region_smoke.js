// SPANA engine v2-rebuild — REBUILT 2026-10 — NOT ORIGINAL SOURCE
// Browser smoke test of tools_rb/r2_jersey_region_annotator.html on a SYNTHETIC pack (driven by
// test_r2_jersey_region_browser.py). usage: node jersey_region_smoke.js PACK_DIR OUT_JSON
const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1700,height:1100},acceptDownloads:true});
const errs=[];p.on('pageerror',e=>errs.push(String(e)));p.on('console',m=>{if(m.type()==='error')errs.push(m.text())});
p.on('dialog',d=>{errs.push('dialog: '+d.message());d.dismiss();});
await p.goto('file://'+process.argv[2]+'/regions.html');
await p.evaluate(()=>localStorage.clear());await p.reload();
await p.fill('#annotator','smoke');
await p.fill('#pSn','7');
const v1=await p.evaluate(()=>v1);
const kfs=await p.evaluate(()=>clip.keyframes);
const st=async(row,state)=>p.click(`#regTable tr:nth-child(${row}) button[data-v="${state}"]`);
for(let i=0;i<kfs.length;i++){const k=v1.keyframes[String(kfs[i])];
  if(k.visibility==='VISIBLE'||k.visibility==='PARTIAL'){
    if(k.number==='READABLE'){await st(1,'READABLE');await p.fill('#regTable tr:nth-child(1) input >> nth=0','7');}
    else{await st(1,'UNREADABLE');await p.selectOption('#regTable tr:nth-child(1) select','ANGLE');}
    await st(2,'NOT_VISIBLE');
    await st(3,'READABLE');await p.fill('#regTable tr:nth-child(3) input >> nth=0','7');
    await st(4,'NO_NUMBER');}
  await p.keyboard.press('ArrowRight');}
await p.click('#strip button >> nth=0');
await p.fill('#cDesc','same kit right');await p.selectOption('#cRel','SAME_TEAM');await p.fill('#cJn','8');await p.click('#bAddC');
await p.click('#cVis button[data-v="VISIBLE"]');
const cv=await p.$('#cv');const bb=await cv.boundingBox();
await p.mouse.move(bb.x+bb.width*0.4,bb.y+bb.height*0.2);await p.mouse.down();await p.mouse.move(bb.x+bb.width*0.45,bb.y+bb.height*0.5,{steps:4});await p.mouse.up();
await st(1,'READABLE');await p.fill('#regTable tr:nth-child(1) input >> nth=0','8');
await st(2,'NOT_VISIBLE');await st(3,'UNREADABLE');await p.selectOption('#regTable tr:nth-child(3) select','TOO_SMALL');await st(4,'NOT_VISIBLE');
await p.keyboard.press('ArrowRight');
await p.click('#cVis button[data-v="OUT_OF_FRAME"]');
await p.click('#bDisMode');await p.selectOption('#dKind','BOARD');await p.fill('#dText','9');
await p.mouse.move(bb.x+bb.width*0.8,bb.y+bb.height*0.1);await p.mouse.down();await p.mouse.move(bb.x+bb.width*0.9,bb.y+bb.height*0.2,{steps:3});await p.mouse.up();
await p.click('#bCheck');const check=await p.textContent('#msgs');
const [d]=await Promise.all([p.waitForEvent('download'),p.click('#btnSave')]);await d.saveAs(process.argv[3]);
console.log(JSON.stringify({check,errs}));await b.close();})();
