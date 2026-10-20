"""
End-to-End User Simulation Test for AI Document Intelligence Assistant.
"""

import sys
import json
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"


def main():
    print("=" * 70)
    print("STEP 1 & 2: VERIFYING WEB UI HTML & STATUS ENDPOINT")
    print("=" * 70)

    # Verify home page
    r_home = requests.get(BASE_URL, timeout=10)
    print(f"GET {BASE_URL}/ -> HTTP {r_home.status_code}")
    assert r_home.status_code == 200, "Home page failed to return HTTP 200"
    assert "AI Document Intelligence & RAG Assistant" in r_home.text, "Title missing from UI HTML"
    print("UI HTML served successfully and contains all interface elements.\n")

    # Verify status endpoint
    r_status = requests.get(f"{BASE_URL}/api/status", timeout=10)
    status_data = r_status.json()
    print(f"GET {BASE_URL}/api/status -> HTTP {r_status.status_code}")
    print(f"Status Payload: {status_data}\n")
    assert r_status.status_code == 200, "Status endpoint failed"

    print("=" * 70)
    print("STEP 3: SIMULATING USER UPLOAD OF test.pdf.pdf")
    print("=" * 70)

    with open("test.pdf.pdf", "rb") as f:
        pdf_bytes = f.read()

    upload_headers = {"X-Filename": "test.pdf.pdf"}
    r_upload = requests.post(
        f"{BASE_URL}/api/upload",
        headers=upload_headers,
        data=pdf_bytes,
        timeout=60,
    )
    print(f"POST {BASE_URL}/api/upload -> HTTP {r_upload.status_code}")
    upload_res = r_upload.json()
    print(f"Upload Response: {json.dumps(upload_res, indent=2)}")
    assert r_upload.status_code == 200, "Upload request failed"
    assert upload_res.get("success") is True, "Upload reported failure"
    print("PDF uploaded and ingested successfully!\n")

    print("=" * 70)
    print("STEP 4 & 5: ASKING TARGET QUESTION")
    print("=" * 70)

    question_1 = "What is the main objective of this research paper?"
    print(f'User Query: "{question_1}"\n')

    r_q1 = requests.post(
        f"{BASE_URL}/api/query",
        json={"question": question_1, "top_k": 3},
        timeout=30,
    )
    print(f"POST {BASE_URL}/api/query -> HTTP {r_q1.status_code}")
    res1 = r_q1.json()
    assert r_q1.status_code == 200, "Query request failed"
    assert res1.get("success") is True, "Query reported failure"

    print("\n[GENERATED ANSWER]")
    print("-" * 70)
    print(res1["answer"])
    print("-" * 70)

    print("\n[CITATIONS]")
    for c in res1.get("citations", []):
        print(f"  * Document: {c['source']} | Page: {c['page']}")

    print("\n[RETRIEVED CONTEXT CHUNKS]")
    for i, chk in enumerate(res1.get("source_chunks", []), start=1):
        score = chk.get("score", 0.0)
        print(f"  Chunk #{i} [Page {chk.get('page')}, {chk.get('source')}] (L2 Distance: {score:.4f})")
        print(f"    Preview: {chk.get('text', '')[:140]}...")

    # Assertions for Question 1
    assert len(res1.get("citations", [])) > 0, "No citations generated"
    assert any(c["source"] == "test.pdf.pdf" for c in res1["citations"]), "test.pdf.pdf citation missing"
    assert len(res1["answer"]) > 50, "Answer too short"

    print("\n" + "=" * 70)
    print("STEP 6: TESTING OUT-OF-CONTEXT UNANSWERABLE QUESTION")
    print("=" * 70)

    question_2 = "What is the secret recipe for baking traditional Italian tiramisu?"
    print(f'User Query: "{question_2}"\n')

    r_q2 = requests.post(
        f"{BASE_URL}/api/query",
        json={"question": question_2, "top_k": 3},
        timeout=30,
    )
    print(f"POST {BASE_URL}/api/query -> HTTP {r_q2.status_code}")
    res2 = r_q2.json()
    assert r_q2.status_code == 200, "Query request failed"

    print("\n[GENERATED ANSWER]")
    print("-" * 70)
    print(res2["answer"])
    print("-" * 70)

    no_info_phrase = "The provided documents do not contain sufficient information to answer this question."
    declined_properly = no_info_phrase in res2["answer"]
    print(f"\nRefusal Verification: {declined_properly}")
    assert declined_properly, "System failed to refuse unanswerable query!"

    print("\n" + "=" * 70)
    print("ALL USER PERSPECTIVE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
