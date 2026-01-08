"""
Interactive 3D UMAP visualization with topic clusters.
Based on create_interactive_umap_v2_topics.py.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from litellm import acompletion

from .config import Config


def c_tf_idf(documents_per_topic, m, ngram_range=(1, 2)):
    """Class-based TF-IDF for topic keyword extraction."""
    count = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        max_features=5000
    ).fit(documents_per_topic)
    t = count.transform(documents_per_topic).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w, where=w != 0)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t + 1)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


async def get_topic_name(keywords: str, model_id: str, topic: str) -> str:
    """Generate a short topic label using an LLM."""
    prompt = f"""Given these keywords from a cluster of sentences about {topic}:
Keywords: {keywords}

Generate a SHORT (2-4 word) descriptive label for this topic cluster.
Return ONLY the label, nothing else."""

    try:
        max_tok = 1000 if "gpt-5" in model_id else 100

        response = await acompletion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            timeout=90,
            max_tokens=max_tok
        )
        content = response.choices[0].message.content
        if content:
            return content.strip().strip('"').strip("'")[:30]
        return keywords.split(",")[0].strip()[:20]
    except Exception as e:
        print(f"   Error with {model_id}: {e}")
        return keywords.split(",")[0].strip()[:20]


async def generate_topic_names(topic_keywords: dict, config: Config) -> dict:
    """Generate topic names using all configured models."""
    topic_names = {m.name.lower(): {} for m in config.models}

    for topic_id, keywords in topic_keywords.items():
        tid = int(topic_id)
        if tid == -1:
            for m in config.models:
                topic_names[m.name.lower()][-1] = "Outliers"
            continue

        print(f"   Topic {tid}: {keywords[:50]}...")

        tasks = [
            get_topic_name(keywords, m.model_id, config.topic)
            for m in config.models
        ]
        results = await asyncio.gather(*tasks)

        for model, name in zip(config.models, results):
            topic_names[model.name.lower()][tid] = name
            print(f"      {model.name}: {name}")

        await asyncio.sleep(1)  # Rate limit buffer

    return topic_names


def create_interactive_umap(
    df: pd.DataFrame,
    config: Config,
    url_mapping: dict = None
) -> str:
    """
    Create interactive 3D UMAP HTML visualization.

    Args:
        df: DataFrame with text and *_score columns
        config: Config object
        url_mapping: Optional dict mapping article_title -> url

    Returns:
        Path to saved HTML file
    """
    print("1. Generating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(df['text'].tolist(), show_progress_bar=True)

    print("2. Running PCA...")
    pca = PCA(n_components=min(50, len(df) - 1))
    reduced_pca = pca.fit_transform(embeddings)

    print("3. Running UMAP...")
    umap_model = UMAP(n_neighbors=10, n_components=3, min_dist=0.05, metric='cosine', random_state=42)
    coords = umap_model.fit_transform(reduced_pca)

    print("4. Running HDBSCAN clustering...")
    hdbscan = HDBSCAN(min_cluster_size=8, min_samples=3, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = hdbscan.fit_predict(coords)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"   Found {n_clusters} clusters + outliers")

    print("5. Extracting keywords with c-TF-IDF...")
    docs_df = pd.DataFrame({'Doc': df['text'].tolist(), 'Topic': cluster_labels})
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic['Doc'].values, m=len(df))
    words = count.get_feature_names_out()
    labels_list = docs_per_topic['Topic'].values

    topic_keywords = {}
    for i, label in enumerate(labels_list):
        if label == -1:
            topic_keywords[-1] = "outliers, misc"
            continue
        top_indices = tf_idf[:, i].argsort()[-8:][::-1]
        topic_keywords[label] = ", ".join([words[j] for j in top_indices])

    print("6. Generating AI topic names...")
    topic_names = asyncio.run(generate_topic_names(topic_keywords, config))

    print("7. Calculating cluster statistics...")
    df = df.copy()
    df['cluster'] = cluster_labels

    # Add URL if mapping provided
    if url_mapping:
        df['url'] = df['article_title'].map(url_mapping).fillna('')
    elif 'url' not in df.columns:
        df['url'] = ''

    # Get model names for score columns
    model_names = [m.name for m in config.models]

    cluster_stats = {}
    for cluster_id in set(cluster_labels):
        cid = int(cluster_id)
        cluster_df = df[df['cluster'] == cluster_id]
        cluster_coords = coords[cluster_labels == cluster_id]

        stats = {
            'centroid': {
                'x': float(np.mean(cluster_coords[:, 0])),
                'y': float(np.mean(cluster_coords[:, 1])),
                'z': float(np.mean(cluster_coords[:, 2]))
            },
            'count': len(cluster_df),
            'keywords': topic_keywords.get(cluster_id, "")
        }

        for name in model_names:
            col = f'{name}_score'
            if col in cluster_df.columns:
                stats[f'{name.lower()}_mean'] = float(cluster_df[col].mean())

        cluster_stats[cid] = stats

    print("8. Generating interactive HTML...")

    centroid = {
        'x': float(np.mean(coords[:, 0])),
        'y': float(np.mean(coords[:, 1])),
        'z': float(np.mean(coords[:, 2]))
    }

    points_data = []
    for i in range(len(df)):
        row = df.iloc[i]
        point = {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]),
            "sentence": str(row['text'])[:500],
            "source": str(row.get('source', 'Unknown')),
            "title": str(row.get('article_title', ''))[:100],
            "url": str(row.get('url', '')),
            "cluster": int(cluster_labels[i]),
            "mean": float(row.get('mean_score', 5))
        }

        for name in model_names:
            col = f'{name}_score'
            if col in row:
                point[name.lower()] = int(row[col])

        points_data.append(point)

    # Generate HTML (condensed version of the working template)
    html = _generate_html_template(
        points_data=points_data,
        cluster_stats=cluster_stats,
        topic_names=topic_names,
        centroid=centroid,
        config=config,
        n_clusters=n_clusters
    )

    output_file = config.output_dir / "interactive_umap.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n✅ Saved: {output_file}")
    return str(output_file)


def _generate_html_template(
    points_data: list,
    cluster_stats: dict,
    topic_names: dict,
    centroid: dict,
    config: Config,
    n_clusters: int
) -> str:
    """Generate the Three.js HTML visualization."""

    model_names = [m.name for m in config.models]
    model_buttons = ""
    for m in config.models:
        model_buttons += f'''
        <button class="model-btn" data-model="{m.name.lower()}">
            <span class="{m.name.lower()}-color">{m.name}</span>
            <span class="model-id">{m.short_model_id}</span>
        </button>'''

    # Color CSS for models
    model_colors = {
        "GPT": "#10a37f",
        "Claude": "#cc785c",
        "Gemini": "#4285f4"
    }
    color_css = ""
    for name, color in model_colors.items():
        color_css += f".{name.lower()}-color {{ color: {color}; }}\n"

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Triple LLM Analysis - {config.topic}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #0a0a0a; font-family: 'Segoe UI', sans-serif; }}
        #title {{ position: absolute; top: 20px; left: 20px; color: #fff; font-size: 20px; font-weight: 600; z-index: 100; }}
        #controls {{ position: absolute; top: 60px; left: 20px; background: rgba(15, 15, 25, 0.95); padding: 18px; border-radius: 12px; color: white; z-index: 100; min-width: 200px; }}
        .model-btn {{ display: block; width: 100%; padding: 10px 12px; margin: 6px 0; border: 2px solid rgba(255,255,255,0.2); border-radius: 8px; background: rgba(255,255,255,0.05); color: white; cursor: pointer; text-align: left; }}
        .model-btn:hover {{ background: rgba(255,255,255,0.15); }}
        .model-btn.active {{ border-color: #ffd700; background: rgba(255, 215, 0, 0.2); }}
        .model-btn span:first-child {{ display: block; font-weight: 600; font-size: 14px; }}
        .model-id {{ display: block; font-size: 10px; opacity: 0.6; margin-top: 2px; font-family: monospace; }}
        #tooltip {{ position: absolute; background: rgba(15, 15, 30, 0.98); color: white; padding: 16px; border-radius: 10px; pointer-events: none; display: none; max-width: 480px; font-size: 12px; z-index: 1000; border: 1px solid rgba(255, 215, 0, 0.3); }}
        #stats-panel {{ position: absolute; bottom: 20px; right: 20px; background: rgba(15, 15, 25, 0.9); padding: 12px; border-radius: 8px; color: white; font-size: 11px; z-index: 100; }}
        #rating-panel {{ position: absolute; bottom: 20px; left: 20px; background: rgba(15, 15, 25, 0.95); padding: 14px 18px; border-radius: 10px; color: white; font-size: 11px; z-index: 100; max-width: 280px; border: 1px solid rgba(255, 215, 0, 0.2); }}
        #rating-panel .title {{ font-size: 11px; font-weight: 600; color: #ffd700; margin-bottom: 8px; text-transform: uppercase; }}
        #rating-panel .question {{ font-size: 13px; margin-bottom: 10px; line-height: 1.4; }}
        #rating-panel .scale {{ font-family: monospace; font-size: 10px; background: rgba(0,0,0,0.3); padding: 8px 10px; border-radius: 4px; border-left: 2px solid #ffd700; }}
        #color-legend {{ position: absolute; top: 60px; right: 20px; background: rgba(15, 15, 25, 0.95); padding: 14px 16px; border-radius: 10px; color: white; z-index: 100; min-width: 160px; border: 1px solid rgba(255, 215, 0, 0.2); }}
        #color-legend .legend-title {{ font-size: 11px; font-weight: 600; color: #ffd700; margin-bottom: 10px; text-transform: uppercase; }}
        #color-legend .gradient-bar {{ height: 16px; border-radius: 4px; background: linear-gradient(to right, rgb(59, 130, 246), rgb(234, 179, 8), rgb(239, 68, 68)); margin-bottom: 6px; }}
        #color-legend .gradient-labels {{ display: flex; justify-content: space-between; font-size: 10px; margin-bottom: 10px; }}
        #color-legend .scale-labels {{ font-size: 9px; opacity: 0.8; line-height: 1.5; }}
        #color-legend .scale-labels div {{ display: flex; justify-content: space-between; gap: 10px; }}
        {color_css}
    </style>
</head>
<body>
    <div id="title">Triple LLM Analysis: {config.topic}</div>
    <div id="controls">
        <div style="font-size: 12px; font-weight: 600; margin-bottom: 10px; color: #ffd700;">COLOR BY MODEL</div>
        {model_buttons}
        <button class="model-btn" data-model="agreement"><span style="color: #a855f7;">Agreement</span></button>
        <div style="margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px;">
            <label style="display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 13px;">
                <input type="checkbox" id="show-labels" checked style="width: 18px; height: 18px;">
                <span>Show Topic Labels</span>
            </label>
        </div>
    </div>
    <div id="tooltip"></div>
    <div id="stats-panel"><strong>n={len(points_data)}</strong> | Hover to inspect | <strong style="color: #ffd700;">Click to open article</strong></div>
    <div id="color-legend">
        <div class="legend-title">Score Legend</div>
        <div class="gradient-bar"></div>
        <div class="gradient-labels">
            <span>1</span>
            <span>5</span>
            <span>10</span>
        </div>
        <div class="scale-labels">
            <div><span>1</span><span>{config.scale_low}</span></div>
            <div><span>5</span><span>{config.scale_mid}</span></div>
            <div><span>10</span><span>{config.scale_high}</span></div>
        </div>
    </div>
    <div id="rating-panel">
        <div class="title">Rating Task</div>
        <div class="question">{config.rating_display["question"]}</div>
        <div class="scale">
            {config.rating_display["scale"]["low"]}<br>
            {config.rating_display["scale"]["mid"]}<br>
            {config.rating_display["scale"]["high"]}
        </div>
    </div>

    <script type="importmap">
    {{ "imports": {{ "three": "https://cdn.jsdelivr.net/npm/three@0.157.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.157.0/examples/jsm/" }} }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const pointsData = {json.dumps(points_data)};
        const clusterStats = {json.dumps(cluster_stats)};
        const topicNames = {json.dumps(topic_names)};
        const dataCentroid = {json.dumps(centroid)};
        const modelNames = {json.dumps([m.lower() for m in model_names])};

        let currentModel = modelNames[0];
        let showLabels = true;

        function getColorForScore(score, model) {{
            if (model === 'agreement') {{
                const t = Math.min(score / 4, 1);
                return new THREE.Color(`rgb(${{Math.floor(t * 239 + (1-t) * 34)}}, ${{Math.floor((1-t) * 197 + t * 68)}}, ${{Math.floor((1-t) * 94 + t * 68)}})`);
            }}
            const t = (score - 1) / 9;
            let r, g, b;
            if (t < 0.5) {{ r = Math.floor(59 + t*2*175); g = Math.floor(130 + t*2*67); b = Math.floor(246 - t*2*152); }}
            else {{ r = Math.floor(234 + (t-0.5)*2*5); g = Math.floor(179 - (t-0.5)*2*111); b = Math.floor(8 + (t-0.5)*2*60); }}
            return new THREE.Color(`rgb(${{r}}, ${{g}}, ${{b}})`);
        }}

        function getScoreForModel(point, model) {{
            if (model === 'agreement') {{
                const scores = modelNames.map(m => point[m] || 5);
                const mean = scores.reduce((a,b) => a+b) / scores.length;
                return Math.sqrt(scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length);
            }}
            return point[model] || point.mean || 5;
        }}

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0a);
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(dataCentroid.x, dataCentroid.y, dataCentroid.z + 12);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(dataCentroid.x, dataCentroid.y, dataCentroid.z);
        controls.enableDamping = true;
        controls.update();

        scene.add(new THREE.AmbientLight(0xffffff, 0.6));

        const pointGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        const pointMeshes = [];

        pointsData.forEach((p, idx) => {{
            const score = getScoreForModel(p, currentModel);
            const color = getColorForScore(score, currentModel);
            const material = new THREE.MeshPhongMaterial({{ color, emissive: color, emissiveIntensity: 0.4 }});
            const mesh = new THREE.Mesh(pointGeometry, material);
            mesh.position.set(p.x, p.y, p.z);
            mesh.userData = {{ ...p, index: idx }};
            scene.add(mesh);
            pointMeshes.push(mesh);
        }});

        const labelSprites = [];

        function createLabels() {{
            labelSprites.forEach(s => scene.remove(s));
            labelSprites.length = 0;

            for (const [clusterId, stats] of Object.entries(clusterStats)) {{
                if (parseInt(clusterId) === -1 || stats.count < 3) continue;
                const name = topicNames[currentModel]?.[clusterId] || `Cluster ${{clusterId}}`;
                const score = stats[currentModel + '_mean'] || 5;

                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                ctx.font = 'bold 48px Arial';
                const textWidth = ctx.measureText(name).width;
                canvas.width = textWidth + 40;
                canvas.height = 70;

                ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
                ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
                ctx.fill();

                const color = getColorForScore(score, currentModel);
                ctx.strokeStyle = `rgb(${{Math.floor(color.r*255)}}, ${{Math.floor(color.g*255)}}, ${{Math.floor(color.b*255)}})`;
                ctx.lineWidth = 4;
                ctx.roundRect(0, 0, canvas.width, canvas.height, 8);
                ctx.stroke();

                ctx.font = 'bold 48px Arial';
                ctx.fillStyle = ctx.strokeStyle;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(name, canvas.width/2, canvas.height/2);

                const texture = new THREE.CanvasTexture(canvas);
                const material = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
                const sprite = new THREE.Sprite(material);
                sprite.scale.set(canvas.width / 150, canvas.height / 150, 1);
                sprite.position.set(stats.centroid.x, stats.centroid.y + 0.5, stats.centroid.z);
                sprite.visible = showLabels;
                scene.add(sprite);
                labelSprites.push(sprite);
            }}
        }}

        createLabels();

        function updateColors(model) {{
            currentModel = model;
            pointMeshes.forEach((mesh, idx) => {{
                const score = getScoreForModel(pointsData[idx], model);
                const color = getColorForScore(score, model);
                mesh.material.color = color;
                mesh.material.emissive = color;
            }});
            createLabels();
            document.querySelectorAll('.model-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.model === model));
        }}

        document.querySelectorAll('.model-btn').forEach(btn => btn.addEventListener('click', () => updateColors(btn.dataset.model)));
        document.getElementById('show-labels').addEventListener('change', (e) => {{ showLabels = e.target.checked; labelSprites.forEach(s => s.visible = showLabels); }});

        // Set first model active
        document.querySelector('.model-btn').classList.add('active');

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');
        let hoveredMesh = null;

        window.addEventListener('mousemove', (event) => {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(pointMeshes);

            if (intersects.length > 0) {{
                const mesh = intersects[0].object;
                if (hoveredMesh && hoveredMesh !== mesh) hoveredMesh.scale.set(1, 1, 1);
                mesh.scale.set(2.5, 2.5, 2.5);
                hoveredMesh = mesh;

                const d = mesh.userData;
                const scoresHtml = modelNames.map(m => `<div style="flex:1;text-align:center;padding:8px;background:rgba(255,255,255,0.05);border-radius:6px;"><div style="font-size:10px;opacity:0.8;">${{m.toUpperCase()}}</div><div style="font-size:20px;font-weight:bold;">${{d[m] || '?'}}</div></div>`).join('');

                tooltip.innerHTML = `<div style="padding:10px;background:rgba(255,255,255,0.05);border-radius:6px;margin-bottom:12px;border-left:3px solid #ffd700;">${{d.sentence}}</div><div style="display:flex;gap:10px;">${{scoresHtml}}</div><div style="font-size:10px;opacity:0.6;margin-top:6px;">Source: ${{d.source}}</div>${{d.url ? '<div style="margin-top:6px;color:#ffd700;font-size:10px;text-align:center;">Click to open article</div>' : ''}}`;
                tooltip.style.display = 'block';
                tooltip.style.left = (event.clientX + 20) + 'px';
                tooltip.style.top = (event.clientY + 20) + 'px';
            }} else {{
                if (hoveredMesh) {{ hoveredMesh.scale.set(1, 1, 1); hoveredMesh = null; }}
                tooltip.style.display = 'none';
            }}
        }});

        window.addEventListener('click', (event) => {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(pointMeshes);
            if (intersects.length > 0 && intersects[0].object.userData.url) {{
                window.open(intersects[0].object.userData.url, '_blank');
            }}
        }});

        function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
        window.addEventListener('resize', () => {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }});
        animate();
    </script>
</body>
</html>
"""
