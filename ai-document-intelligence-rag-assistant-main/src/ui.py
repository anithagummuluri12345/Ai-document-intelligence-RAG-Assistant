"""
Web User Interface Component for AI Document Intelligence & RAG Assistant.

Provides a clean, standalone, dependency-free web interface using Python's
built-in HTTP server to upload PDFs, ask questions, and display cited answers.
"""

import os
import sys
import json
import logging
import socket
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Union

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.rag_pipeline import RAGPipeline, DEFAULT_DB_DIR
except ImportError:
    from rag_pipeline import RAGPipeline, DEFAULT_DB_DIR

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Document Intelligence & RAG Assistant</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --success: #10b981;
            --badge-bg: #1e3a8a;
            --badge-text: #bfdbfe;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
            display: flex;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 960px;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        header {
            text-align: center;
            margin-bottom: 0.5rem;
        }

        header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            font-size: 0.875rem;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.2rem 0.6rem;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            font-weight: 500;
        }

        .status-pill.dot::before {
            content: "";
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--surface-border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Upload area */
        .dropzone {
            border: 2px dashed var(--surface-border);
            border-radius: 8px;
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(15, 23, 42, 0.4);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: rgba(37, 99, 235, 0.05);
        }

        .dropzone svg {
            width: 40px;
            height: 40px;
            stroke: var(--accent);
            margin-bottom: 0.5rem;
        }

        .dropzone p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .dropzone span {
            color: var(--accent);
            font-weight: 500;
        }

        input[type="file"] {
            display: none;
        }

        .upload-actions {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
            align-items: center;
        }

        /* Form elements */
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.65rem 1.25rem;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }

        .btn:hover:not(:disabled) {
            background: var(--primary-hover);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #334155;
            color: #e2e8f0;
        }

        .btn-secondary:hover:not(:disabled) {
            background: #475569;
        }

        .query-box {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        textarea {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            resize: vertical;
            min-height: 80px;
            outline: none;
        }

        textarea:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 1px var(--primary);
        }

        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.25rem;
        }

        .sample-pill {
            background: rgba(51, 65, 85, 0.6);
            color: var(--text-muted);
            border: 1px solid var(--surface-border);
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.775rem;
            cursor: pointer;
            transition: all 0.15s;
        }

        .sample-pill:hover {
            color: var(--accent);
            border-color: var(--accent);
            background: rgba(56, 189, 248, 0.08);
        }

        /* Results Display */
        .result-container {
            display: none;
            flex-direction: column;
            gap: 1.25rem;
            margin-top: 0.5rem;
        }

        .answer-box {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--surface-border);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 1.25rem;
            line-height: 1.6;
            white-space: pre-wrap;
            font-size: 0.95rem;
        }

        .citations-section {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .citations-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .citation-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .badge {
            background: var(--badge-bg);
            color: var(--badge-text);
            border: 1px solid rgba(147, 197, 253, 0.3);
            border-radius: 6px;
            padding: 0.35rem 0.75rem;
            font-size: 0.825rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .chunks-details {
            border: 1px solid var(--surface-border);
            border-radius: 8px;
            overflow: hidden;
        }

        .chunks-summary {
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.5);
            font-size: 0.85rem;
            color: var(--text-muted);
            cursor: pointer;
            user-select: none;
            font-weight: 500;
        }

        .chunk-list {
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            background: rgba(15, 23, 42, 0.3);
        }

        .chunk-item {
            border-left: 2px solid var(--surface-border);
            padding-left: 0.75rem;
            font-size: 0.85rem;
            color: #cbd5e1;
            line-height: 1.5;
        }

        .chunk-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.2rem;
            font-weight: 600;
        }

        .spinner {
            display: none;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .msg-box {
            display: none;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }

        .msg-box.success {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .msg-box.error {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Document Intelligence & RAG Assistant</h1>
            <p>Page-Level Citation Tracking & Grounded Document Question Answering</p>
        </header>

        <div class="status-bar">
            <span>RAG Core Pipeline</span>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <span id="indexedCount" style="color: var(--text-muted);">Indexed Chunks: ...</span>
                <span class="status-pill dot">Ready</span>
            </div>
        </div>

        <!-- Document Ingestion Card -->
        <div class="card">
            <div class="card-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                Document Ingestion
            </div>

            <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                <p>Click or drag & drop a PDF file here</p>
                <p id="fileLabel" style="font-size: 0.8rem; margin-top: 0.25rem;">Supports PDF documents</p>
            </div>
            <input type="file" id="fileInput" accept=".pdf">

            <div class="upload-actions">
                <button class="btn" id="uploadBtn" disabled onclick="uploadSelectedFile()">
                    <span class="spinner" id="uploadSpinner"></span>
                    <span>Upload & Ingest PDF</span>
                </button>
                <button class="btn btn-secondary" id="sampleBtn" onclick="ingestExistingSample()">
                    <span>Ingest test.pdf.pdf</span>
                </button>
            </div>
            <div id="uploadMsg" class="msg-box"></div>
        </div>

        <!-- Query Card -->
        <div class="card">
            <div class="card-title">
                <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                Ask Document Assistant
            </div>

            <div class="query-box">
                <textarea id="questionInput" placeholder="Enter your question based on the ingested document (e.g., 'What architecture is proposed to detect steganographic threats?')"></textarea>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="samples">
                        <span class="sample-pill" onclick="setQuery(this.innerText)">What classifier is proposed to detect steganographic threats in images?</span>
                        <span class="sample-pill" onclick="setQuery(this.innerText)">Why are Deep Neural Networks good candidates to reveal hidden threats?</span>
                        <span class="sample-pill" onclick="setQuery(this.innerText)">What is the capital of Australia?</span>
                    </div>
                </div>

                <div style="margin-top: 0.5rem;">
                    <button class="btn" id="queryBtn" onclick="submitQuestion()">
                        <span class="spinner" id="querySpinner"></span>
                        <span>Submit Question</span>
                    </button>
                </div>
            </div>

            <div id="queryMsg" class="msg-box"></div>

            <!-- Result Section -->
            <div class="result-container" id="resultContainer">
                <div>
                    <div style="font-size: 0.875rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.4rem;">GENERATED ANSWER</div>
                    <div class="answer-box" id="answerText"></div>
                </div>

                <div class="citations-section" id="citationsSection">
                    <div class="citations-title">Verified Citations</div>
                    <div class="citation-badges" id="citationBadges"></div>
                </div>

                <details class="chunks-details">
                    <summary class="chunks-summary">View Retrieved Source Chunks & Distance Scores</summary>
                    <div class="chunk-list" id="chunksList"></div>
                </details>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;

        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const fileLabel = document.getElementById('fileLabel');
        const uploadBtn = document.getElementById('uploadBtn');

        // Drag and drop handlers
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.name.toLowerCase().endsWith('.pdf')) {
                showMessage('uploadMsg', 'Please select a valid PDF file.', 'error');
                return;
            }
            selectedFile = file;
            fileLabel.innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
            uploadBtn.disabled = false;
            hideMessage('uploadMsg');
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('indexedCount').innerText = `Indexed Chunks: ${data.indexed_chunks}`;
            } catch (err) {
                console.error('Failed to fetch status:', err);
            }
        }

        async function uploadSelectedFile() {
            if (!selectedFile) return;

            const btn = document.getElementById('uploadBtn');
            const spinner = document.getElementById('uploadSpinner');
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            hideMessage('uploadMsg');

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    headers: {
                        'X-Filename': selectedFile.name,
                    },
                    body: selectedFile
                });
                const resData = await response.json();
                if (response.ok && resData.success) {
                    showMessage('uploadMsg', `Successfully ingested '${resData.filename}'! Indexed ${resData.chunks_indexed} chunks. (Total in DB: ${resData.total_chunks})`, 'success');
                    fetchStatus();
                } else {
                    showMessage('uploadMsg', resData.error || 'Failed to ingest file.', 'error');
                }
            } catch (err) {
                showMessage('uploadMsg', 'Upload failed: ' + err.message, 'error');
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        async function ingestExistingSample() {
            const btn = document.getElementById('sampleBtn');
            btn.disabled = true;
            hideMessage('uploadMsg');

            try {
                const response = await fetch('/api/ingest_existing', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ filename: 'test.pdf.pdf' })
                });
                const resData = await response.json();
                if (response.ok && resData.success) {
                    showMessage('uploadMsg', `Successfully ingested sample 'test.pdf.pdf'! Indexed ${resData.chunks_indexed} chunks. (Total in DB: ${resData.total_chunks})`, 'success');
                    fetchStatus();
                } else {
                    showMessage('uploadMsg', resData.error || 'Failed to ingest sample.', 'error');
                }
            } catch (err) {
                showMessage('uploadMsg', 'Sample ingestion failed: ' + err.message, 'error');
            } finally {
                btn.disabled = false;
            }
        }

        function setQuery(text) {
            document.getElementById('questionInput').value = text;
        }

        async function submitQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            if (!question) {
                showMessage('queryMsg', 'Please enter a question.', 'error');
                return;
            }

            const btn = document.getElementById('queryBtn');
            const spinner = document.getElementById('querySpinner');
            const resultContainer = document.getElementById('resultContainer');
            
            btn.disabled = true;
            spinner.style.display = 'inline-block';
            hideMessage('queryMsg');

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question, top_k: 3 })
                });
                const data = await response.json();

                if (response.ok && data.success) {
                    // Display answer
                    document.getElementById('answerText').innerText = data.answer;

                    // Display citation badges
                    const badgesContainer = document.getElementById('citationBadges');
                    badgesContainer.innerHTML = '';
                    if (data.citations && data.citations.length > 0) {
                        data.citations.forEach(c => {
                            const badge = document.createElement('span');
                            badge.className = 'badge';
                            badge.innerHTML = `&#128196; ${c.source} &bull; Page ${c.page}`;
                            badgesContainer.appendChild(badge);
                        });
                        document.getElementById('citationsSection').style.display = 'flex';
                    } else {
                        document.getElementById('citationsSection').style.display = 'none';
                    }

                    // Display source chunks
                    const chunksList = document.getElementById('chunksList');
                    chunksList.innerHTML = '';
                    if (data.source_chunks) {
                        data.source_chunks.forEach((chunk, i) => {
                            const item = document.createElement('div');
                            item.className = 'chunk-item';
                            item.innerHTML = `
                                <div class="chunk-meta">Chunk #${i + 1} &bull; Document: ${chunk.source} &bull; Page ${chunk.page} (L2 Distance: ${chunk.score ? chunk.score.toFixed(4) : 'N/A'})</div>
                                <div>${chunk.text}</div>
                            `;
                            chunksList.appendChild(item);
                        });
                    }

                    resultContainer.style.display = 'flex';
                } else {
                    showMessage('queryMsg', data.error || 'Failed to retrieve answer.', 'error');
                }
            } catch (err) {
                showMessage('queryMsg', 'Query error: ' + err.message, 'error');
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        // Allow Enter key in query textarea (Shift+Enter for newline)
        document.getElementById('questionInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitQuestion();
            }
        });

        function showMessage(elementId, text, type) {
            const el = document.getElementById(elementId);
            el.className = `msg-box ${type}`;
            el.innerText = text;
            el.style.display = 'block';
        }

        function hideMessage(elementId) {
            const el = document.getElementById(elementId);
            el.style.display = 'none';
        }

        // Initial status load
        fetchStatus();
    </script>
</body>
</html>
"""


class RAGHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving web UI and REST endpoints for RAG pipeline.
    """
    pipeline: Optional[RAGPipeline] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default verbose access logging to keep console clean."""
        logger.info("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        """Helper to send JSON responses."""
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json_body(self) -> Dict[str, Any]:
        """Helper to read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        """Route GET requests."""
        if self.path in ("/", "/index.html"):
            html_bytes = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        elif self.path == "/api/status":
            chunks = self.pipeline.retriever.vector_db.ntotal if self.pipeline else 0
            self.send_json(200, {
                "status": "ready",
                "indexed_chunks": chunks,
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        """Route POST requests."""
        if self.path == "/api/upload":
            content_length = int(self.headers.get("Content-Length", 0))
            filename = self.headers.get("X-Filename", "uploaded_doc.pdf")
            safe_filename = Path(filename).name

            target_dir = Path("data")
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / safe_filename

            # Read uploaded PDF stream
            pdf_bytes = self.rfile.read(content_length)
            target_file.write_bytes(pdf_bytes)

            try:
                chunks_indexed = self.pipeline.ingest_pdf(target_file)
                total_chunks = self.pipeline.retriever.vector_db.ntotal
                self.send_json(200, {
                    "success": True,
                    "filename": safe_filename,
                    "chunks_indexed": chunks_indexed,
                    "total_chunks": total_chunks,
                })
            except Exception as exc:
                logger.exception("Ingestion failed for %s", safe_filename)
                self.send_json(500, {"success": False, "error": str(exc)})

        elif self.path == "/api/ingest_existing":
            data = self.read_json_body()
            filename = data.get("filename", "test.pdf.pdf")
            target_file = Path(filename)
            if not target_file.exists():
                target_file = Path("data") / filename

            if not target_file.exists():
                self.send_json(404, {"success": False, "error": f"File '{filename}' not found."})
                return

            try:
                chunks_indexed = self.pipeline.ingest_pdf(target_file)
                total_chunks = self.pipeline.retriever.vector_db.ntotal
                self.send_json(200, {
                    "success": True,
                    "filename": target_file.name,
                    "chunks_indexed": chunks_indexed,
                    "total_chunks": total_chunks,
                })
            except Exception as exc:
                logger.exception("Ingestion failed for %s", filename)
                self.send_json(500, {"success": False, "error": str(exc)})

        elif self.path == "/api/query":
            data = self.read_json_body()
            question = data.get("question", "").strip()

            if not question:
                self.send_json(400, {"success": False, "error": "Question is required."})
                return

            top_k = data.get("top_k", 3)
            try:
                res = self.pipeline.query(question, top_k=top_k)
                self.send_json(200, {
                    "success": True,
                    "question": question,
                    "answer": res["answer"],
                    "citations": res["citations"],
                    "source_chunks": res["source_chunks"],
                })
            except Exception as exc:
                logger.exception("Query failed: %s", question)
                self.send_json(500, {"success": False, "error": str(exc)})

        else:
            self.send_response(404)
            self.end_headers()


def find_free_port(start_port: int = 8000, max_attempts: int = 20) -> int:
    """Finds an available local TCP port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def start_server(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    db_dir: Union[str, Path] = DEFAULT_DB_DIR,
) -> ThreadingHTTPServer:
    """
    Initializes the RAG pipeline and starts the HTTP server.
    """
    if port is None:
        port = find_free_port(start_port=8000)

    pipeline = RAGPipeline(db_dir=db_dir)
    RAGHTTPRequestHandler.pipeline = pipeline

    server = ThreadingHTTPServer((host, port), RAGHTTPRequestHandler)
    print(f"\n[+] AI Document Intelligence Assistant running at: http://{host}:{port}")
    print(f"[+] Vector database connected: '{db_dir}' ({pipeline.retriever.vector_db.ntotal} chunks indexed)")
    print(f"[+] Press Ctrl+C to stop the server.\n")

    return server


__all__ = ["start_server", "RAGHTTPRequestHandler", "HTML_TEMPLATE"]


if __name__ == "__main__":
    server = start_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server...")
        server.server_close()
