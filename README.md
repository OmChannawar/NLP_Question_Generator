# 📚 QuizForge – Automatic Question Generation from Textbooks

An NLP project that reads a textbook (**PDF or Word**, max **5 MB**) and generates exam-style questions at a chosen
**difficulty (Low / Mid / High)** and **quantity (5, 10, 15, 20, 25 or 30)**. After generation, an
**"Answer the questions"** button reveals the **keywords of each answer** (plus the answer and its source sentence).

## 1. Key NLP / AI concepts

| Concept | Where it is used |
|---|---|
| **Seq2Seq** | Flan-T5 is an *encoder–decoder* model: the encoder reads *instruction + answer + passage*, the decoder writes the question token by token (`qgen/generator.py`). |
| **Transformers** | Both encoder and decoder are Transformer stacks (self-/cross-attention), loaded through Hugging Face `transformers` (`AutoModelForSeq2SeqLM`). |
| **LLM** | Flan-T5 is an *instruction-tuned* language model. The difficulty is controlled by **prompt engineering** (a different instruction per level, Bloom's-taxonomy style). |

Classical NLP surrounds the neural core: layout/heading detection, sentence splitting, **TF-IDF** sentence scoring and keyword extraction.

## 2. How it works (answer-aware question generation)

```
PDF / DOCX ─► text + line structure ─► section-aware passages (title + text)
                                              │
              pick WHAT to ask: answer candidates (TF-IDF scored) spread over the book
                                              ▼
   Streamlit UI ◄─ keywords (TF-IDF) ◄─ checks ◄─ Flan-T5 writes a question FOR that answer
```

1. **Extraction** – `pypdf` / `python-docx` read the file and keep line breaks.
2. **Section-aware passages** – headings are detected (short lines without final punctuation, Word heading styles);
   bullets/numbered items are re-joined; each passage stays inside one section and carries its title. Syllabus /
   contents / reference sections are skipped.
3. **Answer candidates** (`qgen/candidates.py`) – the *answer* is chosen from the book first:
   * **Low** – a key term (from `Term: definition` or `X is a …` sentences)
   * **Mid** – an informative sentence (explain / describe style)
   * **High** – a sentence with reasoning cues (because, helps, prevents, advantage, balance …)

   Sentences are ranked by document-level TF-IDF; passages are spread evenly over the whole book.
4. **Question generation** – Flan-T5 receives *style hint + answer + passage* and writes the question
   (batched; greedy by default, optional beam search).
5. **Quality checks** – a question is accepted only if (a) its topic words appear in the passage (no hallucinated
   topics), (b) it does not already contain its short answer, (c) it is not a near-duplicate. Otherwise the slot is
   retried with another sentence from the same passage.
6. **Answers & keywords** – the answer is the textbook text itself (it cannot be invented by the model); TF-IDF
   extracts its keywords.

## 3. Tech stack

**Frontend**
* Python **Streamlit** – web UI, widgets (file uploader, select-sliders, progress bar, download button)
* Custom **HTML + CSS** injected via `st.markdown` (gradient header, question cards, keyword chips)

**Backend**
* **Python 3.9+**
* **PyTorch** – tensor / GPU computation (bfloat16 on NVIDIA GPUs)
* **Hugging Face Transformers** + **SentencePiece** – Flan-T5 (Small / Base / Large) Seq2Seq LLM and tokenizer
* **scikit-learn** – TF-IDF scoring and keyword extraction
* **NumPy** – even sampling of passages
* **pypdf** and **python-docx** – PDF and Word parsing

## 4. Project structure

```
question-generator/
├── app.py                  # Streamlit UI
├── requirements.txt        # all dependencies
├── README.md               # this documentation
├── .gitignore              # files excluded from Git
├── .streamlit/config.toml  # server config (5 MB upload limit)
├── notebooks/
│   └── question_generation_demo.ipynb   # step-by-step demo of the pipeline (no UI)
└── qgen/
    ├── extractor.py        # file reading, heading detection, section-aware chunking
    ├── candidates.py       # chooses the answers (terms / sentences) the questions will be about
    ├── generator.py        # Seq2Seq LLM: prompts, batched generation, quality checks
    ├── keywords.py         # TF-IDF keyword extraction
    └── pipeline.py         # end-to-end orchestration
```

## 5. Installation & run

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows:  venv\Scripts\activate

# 2. install every requirement
pip install -r requirements.txt
#    NVIDIA GPU? install a CUDA build of PyTorch first: https://pytorch.org/get-started/locally/

# 3. start the app (run from the project folder so .streamlit/config.toml is used)
streamlit run app.py
```
Open http://localhost:8501. The first generation downloads the chosen model (Base ≈ 1 GB) and needs internet once.

## 6. Usage
1. Upload a `.pdf` or `.docx` textbook, **maximum 5 MB** (must contain selectable text – scanned images are not supported).
2. Choose the **difficulty** and the **number of questions** (5–30, step 5).
3. Click **Generate questions**.
4. Click **Answer the questions (show keywords)** to see the keywords, the answer and its source sentence.
5. Optionally download the questions as `.txt`.

## 7. Known limitations (please read)
* Flan-T5 is a small LLM: some questions can still be awkwardly worded, and the difficulty match is approximate
  (it is mostly set by *which sentence* is chosen as the answer). **Flan-T5 Large** and the *Better quality* mode help.
* Quality depends on document structure. Notes/slides/textbooks with headings and `Term: definition` lines work best;
  long unstructured prose, tables, formulas and code listings work worse.
* If the document has too few suitable sentences, fewer questions than requested are returned.
* Heading detection is heuristic (it can mistake a short line for a heading or miss one).
* Speed: GPU strongly recommended (the sidebar shows whether one was detected). On CPU use Flan-T5 Small or Base in Fast mode.
* Uploads are limited to 5 MB. Scanned PDFs need OCR first.

## 8. Possible extensions
Fine-tune T5 on SQuAD/RACE for question generation, round-trip QA verification (answer the question with a QA model and
compare), swap in a larger LLM, add multiple-choice distractors, OCR support, automatic difficulty scoring.
