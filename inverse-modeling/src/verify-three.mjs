import assert from 'node:assert/strict';
import {Vector3} from 'three';
import {createPosterior3D} from './three-view.js';
const events={};
let draws=0,redraws=0;
const fake={setClearColor(){},setPixelRatio(){},setSize(){},render(scene){scene.updateMatrixWorld();draws++;},dispose(){}};
const app=createPosterior3D({canvas:{style:{},addEventListener(k,f){events[k]=f;}},redraw(){redraws++;},rendererFactory:()=>fake});
const cloud=[{u:[-.6,.1,.3],d:.1},{u:[.4,-.8,-.2],d:1},{u:[.8,.6,.7],d:3},{u:[0,0,0],d:7}];
const rgb=()=>[104,111,196];
for(const [w,h] of [[1100,560],[640,460],[390,460]]){
 for(const yaw of [-2,-.74,.3,2])for(const pitch of [-1.2,.48,1.2])for(const zoom of [.65,1,2.6]){
  const scale=Math.min(w*.285,h*.225)*zoom,pose={yaw,pitch,zoom};
  app.render({cloud,ordered:false,level:6,rgb,view:{w,h,dpr:2},pose,pixelsPerUnit:scale});
  const info=app.inspect();assert.equal(info.count,3);
  for(const point of cloud.slice(0,3)){
   const p=point.u,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
   const x=w/2+scale*(cy*p[0]+sy*p[1]);
   const y=h*.49-scale*(cp*p[2]-sp*(-sy*p[0]+cy*p[1]));
   const q=new Vector3(...p).project(info.camera);
   assert(Math.abs((q.x+1)*w/2-x)<1e-8,'Canvas label and Three.js point X must agree');
   assert(Math.abs((1-q.y)*h/2-y)<1e-8,'Canvas label and Three.js point Y must agree');
   if(x>0&&x<w&&y>0&&y<h){const picked=app.pick(x,y,3);assert(picked);assert(picked.every((v,k)=>Math.abs(v-p[k])<1e-6),'Raycaster must return the displayed parameter combination');}
  }
  for(let i=1;i<info.indices.length;i++)assert(info.depths[info.indices[i-1]]>=info.depths[info.indices[i]],'Transparent points must be sorted far to near');
 }
}
app.render({cloud,ordered:true,level:6,rgb,view:{w:390,h:460,dpr:2},pose:{yaw:0,pitch:.48},pixelsPerUnit:90});
assert.equal(app.inspect().count,1,'Ordering must filter component-label duplicates');
events.webglcontextlost({preventDefault(){}});assert.equal(app.isAvailable(),false);
events.webglcontextrestored();assert.equal(app.isAvailable(),true);assert.equal(redraws,2);
assert(draws>100);
console.log('Three.js projection, ray picking, transparency order, display filtering, and context-loss handling passed at desktop and phone dimensions.');
console.log('Rendering was injected; these checks do not measure a physical phone GPU.');
