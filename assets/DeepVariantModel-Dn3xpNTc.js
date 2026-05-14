import{a as g,i as _,m as S,r as P,l as y,z as E,g as b,T as l}from"./index-ChyOf6jv.js";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function p(h,t,e){if(g(h),t!=null&&t.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const r=_(h,e);if(r.length!==4&&r.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(r.length===1&&t==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return S(h,t,r,e)}const i=[100,221,7],f=["hom_ref","het","hom_alt"],m=i[0]*i[1]*i[2];class u{constructor(t,e,r){this.model=t,this.precision=e,this.backend=r}static INPUT_SHAPE=i;static CLASSES=f;static async load(t={}){const e=t.precision??"uint8",d=(t.modelBaseUrl??"/deepvariant-tfjs/models/").replace(/\/?$/,"/"),n=e==="uint8"?"tfjs_dv_wgs_uint8":"tfjs_dv_wgs";await P(),t.onStage?.("fetching");const c=await y(`${d}${n}/model.json`,{onProgress:t.onProgress});t.onStage?.("warming"),await new Promise(o=>setTimeout(o,16));const a=E([1,...i]),s=c.predict(a);return await s.data(),a.dispose(),s.dispose(),t.onStage?.("ready"),new u(c,e,b())}async predict(t){const[e]=await this.predictBatch(this.asBatch1(t));return e}async predictBatch(t){const e=t instanceof l,r=e?t:p(t.data,[t.batch,...i]);try{const d=this.model.predict(r),n=await d.data();d.dispose();const c=[];for(let a=0;a<n.length;a+=3){const s={hom_ref:n[a],het:n[a+1],hom_alt:n[a+2]};let o=0;s.het>s.hom_ref&&(o=1);const w=o===0?s.hom_ref:s.het;s.hom_alt>w&&(o=2),c.push({probs:s,argmax:f[o],confidence:n[a+o]})}return c}finally{e||r.dispose()}}countParams(){return this.model.countParams()}dispose(){this.model.dispose()}asBatch1(t){if(t instanceof l)return t.expandDims(0);if(t.length!==m)throw new Error(`pileup length ${t.length} != expected ${m} (100*221*7)`);return{data:t,batch:1}}}export{f as DV_CLASSES,i as DV_INPUT_SHAPE,u as DeepVariantModel};
