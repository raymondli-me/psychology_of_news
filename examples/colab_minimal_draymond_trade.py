"""
Minimal Colab Script: Draymond Green Trade Analysis
====================================================
Copy each cell into a Colab notebook.
Reproduces the full triple-LLM analysis with interactive 3D UMAP.
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================
# !pip install -q litellm eventregistry sentence-transformers umap-learn hdbscan nltk pandas numpy scikit-learn

# =============================================================================
# CELL 2: API Keys (EDIT THESE!)
# =============================================================================
import os
os.environ["EVENT_REGISTRY_API_KEY"] = "YOUR_EVENT_REGISTRY_KEY"  # eventregistry.org
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-..."
os.environ["GOOGLE_API_KEY"] = "AIzaSy..."

# =============================================================================
# CELL 3: Configuration
# =============================================================================
TOPIC = "Draymond Green trade"
MAX_ARTICLES = 100
MAX_SENTENCES = 200

MODELS = {
    "GPT": "openai/gpt-5-mini",
    "Claude": "anthropic/claude-sonnet-4-5",
    "Gemini": "gemini/gemini-2.5-flash-preview-09-2025"
}

PROMPT = """Rate this sentence on how strongly it implies Draymond Green will be traded.

Sentence: "{text}"

Score from 1-10:
1 = No trade implication at all
5 = Neutral/ambiguous
10 = Strongly implies trade will happen

Reply with ONLY a single number (1-10), nothing else."""

# =============================================================================
# CELL 4: Fetch Articles from Event Registry
# =============================================================================
from eventregistry import EventRegistry, QueryArticlesIter
import pandas as pd

er = EventRegistry(apiKey=os.environ["EVENT_REGISTRY_API_KEY"])
articles = []

concept_uri = er.getConceptUri(TOPIC.split()[0] + " " + TOPIC.split()[1])  # "Draymond Green"
if concept_uri:
    for art in QueryArticlesIter(conceptUri=concept_uri).execQuery(er, maxItems=MAX_ARTICLES):
        articles.append({
            "title": art.get("title"),
            "body_text": art.get("body"),
            "source": art.get("source", {}).get("title"),
            "url": art.get("url")
        })

df_articles = pd.DataFrame(articles)
print(f"Fetched {len(df_articles)} articles")

# =============================================================================
# CELL 5: Extract Sentences
# =============================================================================
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

sentences = []
keyword = "Draymond"

for _, row in df_articles.iterrows():
    text = str(row.get('body_text', '')).replace('\n', ' ')
    for s in nltk.sent_tokenize(text):
        s = s.strip()
        if len(s) > 30 and keyword in s:
            sentences.append({
                "text": s,
                "source": row.get('source', 'Unknown'),
                "article_title": row.get('title', ''),
                "url": row.get('url', '')
            })
            if len(sentences) >= MAX_SENTENCES:
                break
    if len(sentences) >= MAX_SENTENCES:
        break

print(f"Extracted {len(sentences)} sentences containing '{keyword}'")

# =============================================================================
# CELL 6: Rate Sentences with 3 LLMs (Async)
# =============================================================================
import asyncio
import re
import numpy as np
from litellm import acompletion

async def rate_sentence(text, model_name, model_id, semaphore):
    async with semaphore:
        max_tokens = 1000 if "gpt-5" in model_id else 10  # GPT-5-mini needs more for reasoning
        try:
            response = await acompletion(
                model=model_id,
                messages=[{"role": "user", "content": PROMPT.format(text=text)}],
                timeout=60,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content or ""
            match = re.search(r'\b(\d+)\b', content)
            return (model_name, min(max(int(match.group(1)), 1), 10) if match else 5)
        except Exception as e:
            print(f"  {model_name} error: {str(e)[:50]}")
            return (model_name, 5)

async def rate_all():
    semaphores = {name: asyncio.Semaphore(5) for name in MODELS}

    for i, sent in enumerate(sentences):
        if (i + 1) % 20 == 0:
            print(f"Rating {i+1}/{len(sentences)}...")

        tasks = [rate_sentence(sent['text'], name, mid, semaphores[name]) for name, mid in MODELS.items()]
        results = await asyncio.gather(*tasks)

        for name, score in results:
            sent[f'{name}_score'] = score
        sent['mean_score'] = np.mean([sent[f'{m}_score'] for m in MODELS])

await rate_all()
df = pd.DataFrame(sentences)
print(f"\nRated {len(df)} sentences")
print(f"GPT mean: {df['GPT_score'].mean():.2f}, Claude mean: {df['Claude_score'].mean():.2f}, Gemini mean: {df['Gemini_score'].mean():.2f}")

# =============================================================================
# CELL 7: Create Embeddings + UMAP + Clustering
# =============================================================================
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

print("Creating embeddings...")
embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(df['text'].tolist(), show_progress_bar=True)

print("Running PCA + UMAP...")
pca = PCA(n_components=min(50, len(df) - 1))
coords = UMAP(n_neighbors=10, n_components=3, min_dist=0.05, random_state=42).fit_transform(pca.fit_transform(embeddings))

print("Clustering...")
labels = HDBSCAN(min_cluster_size=8, min_samples=3).fit_predict(coords)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} clusters")

# c-TF-IDF keywords
docs_per_topic = pd.DataFrame({'Doc': df['text'], 'Topic': labels}).groupby('Topic').agg({'Doc': ' '.join})
vec = CountVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000).fit(docs_per_topic['Doc'])
tfidf = vec.transform(docs_per_topic['Doc']).toarray()
words = vec.get_feature_names_out()

topic_keywords = {}
for i, tid in enumerate(docs_per_topic.index):
    if tid != -1:
        top_idx = tfidf[i].argsort()[-6:][::-1]
        topic_keywords[int(tid)] = ", ".join([words[j] for j in top_idx])

# =============================================================================
# CELL 8: AI-Generate Topic Names
# =============================================================================
async def get_topic_name(keywords, model_id):
    prompt = f"Given keywords: {keywords}\nGenerate a SHORT (2-4 word) topic label. Return ONLY the label."
    try:
        r = await acompletion(model=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=1000 if "gpt-5" in model_id else 50, timeout=60)
        return (r.choices[0].message.content or keywords.split(",")[0]).strip().strip('"')[:25]
    except:
        return keywords.split(",")[0].strip()[:20]

async def gen_all_names():
    names = {m.lower(): {} for m in MODELS}
    for tid, kw in topic_keywords.items():
        results = await asyncio.gather(*[get_topic_name(kw, mid) for mid in MODELS.values()])
        for m, name in zip(MODELS.keys(), results):
            names[m.lower()][tid] = name
        print(f"Topic {tid}: {list(zip(MODELS.keys(), results))}")
    return names

topic_names = await gen_all_names()

# =============================================================================
# CELL 9: Generate Interactive HTML
# =============================================================================
import json

centroid = {'x': float(coords[:,0].mean()), 'y': float(coords[:,1].mean()), 'z': float(coords[:,2].mean())}

points = [{"x": float(coords[i,0]), "y": float(coords[i,1]), "z": float(coords[i,2]),
           "sentence": df.iloc[i]['text'][:400], "source": str(df.iloc[i]['source']),
           "url": str(df.iloc[i].get('url', '')), "cluster": int(labels[i]),
           "gpt": int(df.iloc[i]['GPT_score']), "claude": int(df.iloc[i]['Claude_score']),
           "gemini": int(df.iloc[i]['Gemini_score']), "mean": float(df.iloc[i]['mean_score'])}
          for i in range(len(df))]

cluster_stats = {}
for cid in set(labels):
    mask = labels == cid
    cluster_stats[int(cid)] = {
        'centroid': {'x': float(coords[mask,0].mean()), 'y': float(coords[mask,1].mean()), 'z': float(coords[mask,2].mean())},
        'count': int(mask.sum()),
        'gpt_mean': float(df.loc[mask, 'GPT_score'].mean()),
        'claude_mean': float(df.loc[mask, 'Claude_score'].mean()),
        'gemini_mean': float(df.loc[mask, 'Gemini_score'].mean())
    }

html = f'''<!DOCTYPE html><html><head><title>Draymond Trade - Triple LLM</title>
<style>body{{margin:0;overflow:hidden;background:#0a0a0a;font-family:sans-serif}}
#title{{position:absolute;top:20px;left:20px;color:#fff;font-size:20px;z-index:100}}
#controls{{position:absolute;top:60px;left:20px;background:rgba(15,15,25,0.95);padding:18px;border-radius:12px;color:#fff;z-index:100}}
.btn{{display:block;width:100%;padding:10px;margin:6px 0;border:2px solid rgba(255,255,255,0.2);border-radius:8px;background:rgba(255,255,255,0.05);color:#fff;cursor:pointer}}
.btn.active{{border-color:#ffd700;background:rgba(255,215,0,0.2)}}
#tooltip{{position:absolute;background:rgba(15,15,30,0.98);color:#fff;padding:16px;border-radius:10px;display:none;max-width:450px;z-index:1000;border:1px solid rgba(255,215,0,0.3)}}
#stats{{position:absolute;bottom:20px;right:20px;background:rgba(15,15,25,0.9);padding:12px;border-radius:8px;color:#fff;font-size:11px;z-index:100}}</style></head>
<body><div id="title">Triple LLM: Draymond Green Trade Analysis</div>
<div id="controls"><div style="font-size:12px;font-weight:600;margin-bottom:10px;color:#ffd700">COLOR BY</div>
<button class="btn active" data-m="gpt">GPT-5-mini (mean={df['GPT_score'].mean():.1f})</button>
<button class="btn" data-m="claude">Claude-4.5 (mean={df['Claude_score'].mean():.1f})</button>
<button class="btn" data-m="gemini">Gemini-2.5 (mean={df['Gemini_score'].mean():.1f})</button>
<button class="btn" data-m="agreement">Agreement</button>
<div style="margin-top:15px;border-top:1px solid rgba(255,255,255,0.1);padding-top:12px">
<label style="display:flex;align-items:center;gap:10px;cursor:pointer"><input type="checkbox" id="labels" checked style="width:18px;height:18px">Show Labels</label></div></div>
<div id="tooltip"></div><div id="stats"><b>n={len(df)}</b> | Hover to inspect | <b style="color:#ffd700">Click for article</b></div>
<script type="importmap">{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.157.0/examples/jsm/"}}}}</script>
<script type="module">
import*as THREE from'three';import{{OrbitControls}}from'three/addons/controls/OrbitControls.js';
const pts={json.dumps(points)},stats={json.dumps(cluster_stats)},names={json.dumps(topic_names)},ctr={json.dumps(centroid)};
let model='gpt',showLbl=true;
const getColor=(s,m)=>{{if(m==='agreement'){{const t=Math.min(s/4,1);return new THREE.Color(`rgb(${{Math.floor(t*239+(1-t)*34)}},${{Math.floor((1-t)*197+t*68)}},${{Math.floor((1-t)*94+t*68)}})`)}}const t=(s-1)/9;return new THREE.Color(`hsl(${{(1-t)*240}},70%,50%)`)}};
const getScore=(p,m)=>m==='agreement'?Math.sqrt([p.gpt,p.claude,p.gemini].reduce((a,s,_,arr)=>a+Math.pow(s-arr.reduce((x,y)=>x+y)/3,2),0)/3):p[m]||p.mean;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0a0a0a);
const cam=new THREE.PerspectiveCamera(75,innerWidth/innerHeight,0.1,1000);cam.position.set(ctr.x,ctr.y,ctr.z+12);
const renderer=new THREE.WebGLRenderer({{antialias:true}});renderer.setSize(innerWidth,innerHeight);document.body.appendChild(renderer.domElement);
const ctrl=new OrbitControls(cam,renderer.domElement);ctrl.target.set(ctr.x,ctr.y,ctr.z);ctrl.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,0.6));
const geo=new THREE.SphereGeometry(0.1,16,16),meshes=[];
pts.forEach((p,i)=>{{const c=getColor(getScore(p,model),model),mat=new THREE.MeshPhongMaterial({{color:c,emissive:c,emissiveIntensity:0.4}}),m=new THREE.Mesh(geo,mat);m.position.set(p.x,p.y,p.z);m.userData={{...p,i}};scene.add(m);meshes.push(m)}});
const sprites=[];
function mkLabels(){{sprites.forEach(s=>scene.remove(s));sprites.length=0;
Object.entries(stats).filter(([id])=>id!=='-1'&&stats[id].count>=3).forEach(([id,st])=>{{
const nm=names[model]?.[id]||'Cluster '+id,sc=st[model+'_mean']||5,cv=document.createElement('canvas'),ctx=cv.getContext('2d');
ctx.font='bold 48px Arial';const tw=ctx.measureText(nm).width;cv.width=tw+40;cv.height=70;
ctx.fillStyle='rgba(0,0,0,0.85)';ctx.roundRect(0,0,cv.width,cv.height,8);ctx.fill();
const col=getColor(sc,model);ctx.strokeStyle=`rgb(${{Math.floor(col.r*255)}},${{Math.floor(col.g*255)}},${{Math.floor(col.b*255)}})`;ctx.lineWidth=4;ctx.roundRect(0,0,cv.width,cv.height,8);ctx.stroke();
ctx.font='bold 48px Arial';ctx.fillStyle=ctx.strokeStyle;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(nm,cv.width/2,cv.height/2);
const tex=new THREE.CanvasTexture(cv),spr=new THREE.Sprite(new THREE.SpriteMaterial({{map:tex,transparent:true,depthTest:false}}));
spr.scale.set(cv.width/150,cv.height/150,1);spr.position.set(st.centroid.x,st.centroid.y+0.5,st.centroid.z);spr.visible=showLbl;scene.add(spr);sprites.push(spr)}})}};mkLabels();
function update(m){{model=m;meshes.forEach((mesh,i)=>{{const c=getColor(getScore(pts[i],m),m);mesh.material.color=c;mesh.material.emissive=c}});mkLabels();document.querySelectorAll('.btn').forEach(b=>b.classList.toggle('active',b.dataset.m===m))}};
document.querySelectorAll('.btn').forEach(b=>b.onclick=()=>update(b.dataset.m));
document.getElementById('labels').onchange=e=>{{showLbl=e.target.checked;sprites.forEach(s=>s.visible=showLbl)}};
const ray=new THREE.Raycaster(),mouse=new THREE.Vector2(),tip=document.getElementById('tooltip');let hov=null;
onmousemove=e=>{{mouse.x=e.clientX/innerWidth*2-1;mouse.y=-e.clientY/innerHeight*2+1;ray.setFromCamera(mouse,cam);
const hit=ray.intersectObjects(meshes);if(hit.length){{const m=hit[0].object;if(hov&&hov!==m)hov.scale.set(1,1,1);m.scale.set(2.5,2.5,2.5);hov=m;const d=m.userData;
tip.innerHTML=`<div style="padding:10px;background:rgba(255,255,255,0.05);border-radius:6px;margin-bottom:12px;border-left:3px solid #ffd700">${{d.sentence}}</div><div style="display:flex;gap:10px"><div style="flex:1;text-align:center;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px"><div style="font-size:10px;color:#10a37f">GPT</div><div style="font-size:20px;font-weight:bold;color:#10a37f">${{d.gpt}}</div></div><div style="flex:1;text-align:center;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px"><div style="font-size:10px;color:#cc785c">Claude</div><div style="font-size:20px;font-weight:bold;color:#cc785c">${{d.claude}}</div></div><div style="flex:1;text-align:center;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px"><div style="font-size:10px;color:#4285f4">Gemini</div><div style="font-size:20px;font-weight:bold;color:#4285f4">${{d.gemini}}</div></div></div><div style="font-size:10px;opacity:0.6;margin-top:8px">Source: ${{d.source}}</div>${{d.url?'<div style="margin-top:6px;color:#ffd700;font-size:10px;text-align:center">Click to open</div>':''}}`
;tip.style.display='block';tip.style.left=(e.clientX+20)+'px';tip.style.top=(e.clientY+20)+'px'}}else{{if(hov){{hov.scale.set(1,1,1);hov=null}};tip.style.display='none'}}}};
onclick=e=>{{mouse.x=e.clientX/innerWidth*2-1;mouse.y=-e.clientY/innerHeight*2+1;ray.setFromCamera(mouse,cam);const hit=ray.intersectObjects(meshes);if(hit.length&&hit[0].object.userData.url)open(hit[0].object.userData.url,'_blank')}};
onresize=()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight)}};
(function anim(){{requestAnimationFrame(anim);ctrl.update();renderer.render(scene,cam)}})();
</script></body></html>'''

with open('/content/draymond_trade_viz.html', 'w') as f:
    f.write(html)
print("Saved: /content/draymond_trade_viz.html")

# =============================================================================
# CELL 10: Display in Colab
# =============================================================================
from IPython.display import HTML, display
display(HTML(html))

# =============================================================================
# CELL 11: Save Data
# =============================================================================
df.to_csv('/content/sentence_ratings.csv', index=False)
print("Saved: /content/sentence_ratings.csv")

# Download files
from google.colab import files
files.download('/content/draymond_trade_viz.html')
files.download('/content/sentence_ratings.csv')
