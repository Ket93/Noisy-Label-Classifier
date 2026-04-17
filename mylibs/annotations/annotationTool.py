# annotations/annotation_tool.py
from flask import Flask, render_template_string, request, jsonify
import os, csv, random
from pathlib import Path
from collections import Counter

app = Flask(__name__)

# --- Configuration ---
BASE_DIR      = Path(__file__).parent.parent
IMAGE_DIR     = str(BASE_DIR / 'data' / 'flowers')
ANNOTATIONS   = str(Path(__file__).parent.parent / 'manual_annotations.csv')
CLASSES       = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
TIME_LIMIT    = 3
CAP_PER_CLASS = 100

# --- Load image list, capped at CAP_PER_CLASS per class ---
all_images = []
for cls in CLASSES:
    folder = Path(IMAGE_DIR) / cls
    imgs   = list(folder.glob('*.jpg'))
    random.shuffle(imgs)
    imgs = imgs[:CAP_PER_CLASS]
    for img in imgs:
        rel = img.relative_to(IMAGE_DIR).as_posix()   # always forward slashes
        all_images.append((rel, cls))

random.shuffle(all_images)

print(f"Looking for images in: {IMAGE_DIR}")
print(f"Exists: {Path(IMAGE_DIR).exists()}")
print(f"Found {len(all_images)} images to annotate ({CAP_PER_CLASS} per class)")

# --- Load already-annotated filenames ---
def get_annotated():
    if not os.path.exists(ANNOTATIONS):
        return {}
    with open(ANNOTATIONS, encoding='utf-8') as f:
        return {row['filename']: row['label'] for row in csv.DictReader(f)}

def get_class_counts(annotated):
    counts = Counter()
    for filename in annotated:
        cls = filename.replace('\\', '/').split('/')[0]
        counts[cls] += 1
    return counts

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Flower Annotator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #1a1a1a;
            color: white;
            padding: 20px;
        }
        img {
            max-width: 500px;
            max-height: 500px;
            border: 3px solid #444;
            border-radius: 8px;
        }
        .timer {
            font-size: 48px;
            font-weight: bold;
            color: #00ff88;
            margin: 15px 0;
            transition: color 0.3s;
        }
        .timer.urgent { color: #ff4444; }
        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
            justify-content: center;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            background: #333;
            color: white;
            transition: background 0.2s;
        }
        button:hover { background: #555; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
        .progress-block {
            margin-bottom: 16px;
            text-align: center;
        }
        .progress-total {
            font-size: 18px;
            color: #aaa;
            margin-bottom: 10px;
        }
        .progress-bars {
            display: flex;
            gap: 16px;
            justify-content: center;
            align-items: flex-end;
        }
        .class-progress {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-size: 11px;
            color: #888;
            width: 60px;
        }
        .bar-bg {
            width: 40px;
            height: 80px;
            background: #333;
            border-radius: 4px;
            display: flex;
            align-items: flex-end;
            margin-bottom: 6px;
        }
        .bar-fill {
            width: 100%;
            background: #00ff88;
            border-radius: 4px;
            transition: height 0.3s;
        }
        .class-name {
            font-size: 11px;
            color: #aaa;
            text-align: center;
            word-break: break-word;
            line-height: 1.3;
        }
        .class-count {
            font-size: 10px;
            color: #666;
        }
        .skip {
            background: #222;
            color: #888;
            font-size: 12px;
            padding: 8px 16px;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="progress-block">
        <div class="progress-total">{{ total_done }} / {{ total_target }} annotated</div>
        <div class="progress-bars">
            {% for cls in classes %}
            <div class="class-progress">
                <div class="bar-bg">
                    <div class="bar-fill" style="height: {{ [((counts.get(cls, 0) / cap) * 100)|int, 100]|min }}%"></div>
                </div>
                <div class="class-name">{{ cls }}</div>
                <div class="class-count">{{ counts.get(cls, 0) }}/{{ cap }}</div>
            </div>
            {% endfor %}
        </div>
    </div>

    <img src="/image/{{ filename }}" id="flower-img">
    <div class="timer" id="timer">{{ time_limit }}</div>

    <div class="buttons" id="btn-group">
        {% for i, cls in classes|enumerate %}
        <button onclick="submitLabel({{ i }})" id="btn-{{ i }}">{{ cls }}</button>
        {% endfor %}
    </div>
    <br>
    <button class="skip" onclick="submitLabel(-1)" id="btn-skip">Skip</button>

    <script>
        const timeLimit  = {{ time_limit }};
        const filename   = {{ filename|tojson }};   // safe JSON encoding, handles backslashes
        const startTime  = Date.now();
        let submitted    = false;

        const timerEl = document.getElementById('timer');

        const interval = setInterval(() => {
            const elapsed   = (Date.now() - startTime) / 1000;
            const remaining = Math.max(0, timeLimit - elapsed).toFixed(1);
            timerEl.textContent = remaining;

            if (remaining < 2) timerEl.classList.add('urgent');

            if (elapsed >= timeLimit && !submitted) {
                clearInterval(interval);
                submitLabel(-1);
            }
        }, 100);

        function submitLabel(label) {
            if (submitted) return;
            submitted = true;
            clearInterval(interval);

            document.querySelectorAll('button').forEach(b => b.disabled = true);
            timerEl.textContent = '✓';
            timerEl.style.color = '#00ff88';

            const timeTaken = (Date.now() - startTime) / 1000;

            fetch('/annotate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename:     filename,
                    label:        label,
                    time_seconds: timeTaken.toFixed(2)
                })
            })
            .then(response => response.json())
            .then(data => {
                window.location.href = '/';
            })
            .catch(err => {
                console.error('Fetch error:', err);
                window.location.href = '/';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    annotated    = get_annotated()
    class_counts = get_class_counts(annotated)

    remaining = [
        (img, cls) for img, cls in all_images
        if img not in annotated and class_counts.get(cls, 0) < CAP_PER_CLASS
    ]

    total_done   = sum(class_counts.get(cls, 0) for cls in CLASSES)
    total_target = CAP_PER_CLASS * len(CLASSES)

    if not remaining:
        return f"""
        <div style='color:white;background:#111;padding:40px;font-family:Arial'>
            <h1>All done! ✓</h1>
            <p>{total_done} / {total_target} images annotated</p>
            <p>Check <b>{ANNOTATIONS}</b> for your results.</p>
            <ul>
                {''.join(f'<li>{cls}: {class_counts.get(cls,0)}</li>' for cls in CLASSES)}
            </ul>
        </div>
        """

    current_img, true_class = remaining[0]

    app.jinja_env.filters['enumerate'] = lambda x: enumerate(x)

    template = app.jinja_env.from_string(HTML)
    return template.render(
        filename=current_img,
        true_class=true_class,
        classes=CLASSES,
        time_limit=TIME_LIMIT,
        cap=CAP_PER_CLASS,
        counts=class_counts,
        total_done=total_done,
        total_target=total_target,
    )

@app.route('/image/<path:filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/annotate', methods=['POST'])
def annotate():
    data       = request.get_json()
    filename   = data['filename']
    label      = data['label']
    time_sec   = data['time_seconds']
    true_class = filename.replace('\\', '/').split('/')[0]

    annotated    = get_annotated()
    class_counts = get_class_counts(annotated)

    if label != -1 and class_counts.get(true_class, 0) < CAP_PER_CLASS:
        file_exists = os.path.exists(ANNOTATIONS)
        with open(ANNOTATIONS, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['filename', 'label', 'time_seconds', 'true_class']
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'filename':     filename,
                'label':        label,
                'time_seconds': time_sec,
                'true_class':   true_class,
            })

    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)