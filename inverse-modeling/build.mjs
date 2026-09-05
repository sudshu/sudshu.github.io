import {build} from 'esbuild';
import {readFile, writeFile} from 'node:fs/promises';
const output = await build({entryPoints:['src/three-view.js'],bundle:true,minify:true,
  format:'iife',platform:'browser',target:['es2020'],write:false,legalComments:'inline'});
const license = await readFile('node_modules/three/LICENSE','utf8');
const code = output.outputFiles[0].text.replace(/<\/script/gi,'<\\/script');
const template = await readFile('src/index.template.html','utf8');
if (!template.includes('<!--THREE_BUNDLE-->')) throw new Error('Missing Three.js insertion point');
await writeFile('index.html',template.replace('<!--THREE_BUNDLE-->',() =>
  '<script id="three-renderer-bundle">\n/*! Three.js 0.185.1\n'+license+'*/\n'+code+'\n</script>'));
console.log('Built offline HTML with embedded Three.js ('+Math.round(code.length/1024)+' KB renderer).');
