# 📚 QuizForge – Automatic Question Generation from Textbooks

An NLP project that reads a textbook (**PDF or Word**) and generates exam-style questions at a chosen
**difficulty (Low / Mid / High)** and **quantity (5, 10, 15, 20, 25 or 30)**. After generation, an
**"Answer the questions"** button reveals the **keywords of each answer**.

## 1. Key NLP / AI concepts

| Concept | Where it is used |
|---|---|
| **Seq2Seq** | Flan-T5 is an *encoder–decoder* model: the encoder reads *instruction + passage*, the decoder generates the question (and answer) one token at a time (`qgen/generator.py`). |
| **Transformers** | Both encoder and decoder are Transformer stacks (self-/cross-attention). Loaded through the Hugging Face `transformers` library (`AutoModelForSeq2SeqLM`). |
| **LLM** | Flan-T5 is an *instruction-tuned* language model. Difficulty is controlled purely by **prompt engineering** – a different natural-language instruction per level (Bloom's-taxonomy style). |

Classical NLP is used around the neural core: regex sentence splitting, text cleaning, and **TF-IDF** keyword extraction.

## 2. How it works

```
PDF / DOCX ──► text extraction ──► cleaning ──► sentence split ──► passages (~170 words)
                                                                        │ evenly sampled across the book
                                                                        ▼
   Streamlit UI  ◄── keywords (TF-IDF) ◄── answer (Flan-T5) ◄── question (Flan-T5 + difficulty prompt)
```

1. **Extraction** – `pypdf` / `python-docx` read the file; text is cleaned (hyphenation, page numbers).
2. **Chunking** – sentences are grouped into short passages that fit the 512-token input window; non-prose
   passages (tables, indexes) are filtered out; N passages are picked evenly across the whole book.
3. **Question generation** – for each passage, Flan-T5 receives a difficulty-specific instruction:
   * **Low** – factual recall (What / Who / When / Where)
   * **Mid** – conceptual understanding (Explain / Describe / How)
   * **High** – analysis & evaluation (Why / Compare / Justify)

   Passages are processed in **batches**. Decoding is greedy by default (fast); an optional beam-search mode
   gives better wording but is slower. Near-duplicate questions are detected on content words and regenerated
   with sampling in later rounds. Answers are then generated in batches too.
4. **Answer + keywords** – the same model answers each question from its passage; TF-IDF ranks the
   answer's terms (IDF learned from the passage sentences) and the top key-phrases are shown as chips.

## 3. Tech stack

**Frontend**
* Python **Streamlit** – web UI, widgets (file uploader, select-sliders, progress bar, download button)
* Custom **HTML + CSS** injected via `st.markdown` (gradient header, question cards, keyword chips)

**Backend**
* **Python 3.9+**
* **PyTorch** – tensor / GPU computation
* **Hugging Face Transformers** + **SentencePiece** – Flan-T5 (Small / Base / Large) Seq2Seq LLM and tokenizer
* **scikit-learn** – TF-IDF keyword extraction
* **NumPy** – even sampling of passages
* **pypdf** and **python-docx** – PDF and Word parsing

## 4. Project structure

```
question-generator/
├── app.py              # Streamlit UI
├── requirements.txt    # all dependencies
├── .streamlit/config.toml  # server config (5 MB upload limit)
├── README.md           # this documentation
├── .gitignore          # files excluded from Git
├── notebooks/
│   └── question_generation_demo.ipynb   # step-by-step demo of the pipeline (no UI)
└── qgen/
    ├── extractor.py    # file reading, cleaning, chunking
    ├── generator.py    # Seq2Seq LLM: prompts, question & answer generation, validation
    ├── keywords.py     # TF-IDF keyword extraction
    └── pipeline.py     # end-to-end orchestration
```

## 5. Installation & run

```bash
# 1. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows:  venv\Scripts\activate

# 2. install every requirement
pip install -r requirements.txt
#    NVIDIA GPU? install a CUDA build of PyTorch first: https://pytorch.org/get-started/locally/

# 3. start the app
streamlit run app.py
```
Open http://localhost:8501. The first generation downloads the chosen model (Base ≈ 1 GB) and needs internet once.

## 6. Usage
1. Upload a `.pdf` or `.docx` textbook, **maximum 5 MB** (must contain selectable text – scanned images are not supported).
2. Choose the **difficulty** and the **number of questions** (5–30, step 5).
3. Click **Generate questions**.
4. Click **Answer the questions (show keywords)** to see the keywords (and the model's suggested answer) per question.
5. Optionally download the questions as `.txt`.

## 7. Known limitations (please read)
* Flan-T5 is a small LLM; some questions may be vague, repetitive or only loosely match the chosen difficulty. **Flan-T5 Large** is noticeably better than Small.
* Answers and keywords are produced by the model from the passage – review them before using in an exam.
* Speed: the app runs the model in bfloat16 on an NVIDIA GPU (float32 on CPU), batches passages, and uses greedy decoding by default.
  A GPU makes by far the biggest difference – the sidebar shows whether one was detected. On CPU choose **Flan-T5 Small** or keep
  **Fast** mode with Base. **Better quality** (beam search) and **Large** are slower. The CUDA build of PyTorch must be installed
  (https://pytorch.org/get-started/locally/) – plain `pip install torch` often gives a CPU-only build on Windows.
* Uploads are limited to 5 MB (`.streamlit/config.toml` and `MAX_UPLOAD_MB` in `app.py`). Run `streamlit run app.py` from the project folder so the config is picked up.
* Scanned PDFs need OCR first. Math-heavy or table-heavy pages are skipped by the prose filter.
* Fewer questions than requested may be returned when the document is short.

## 8. Possible extensions
Fine-tune T5 on SQuAD/RACE for question generation, swap in a larger LLM (e.g. an instruction-tuned Llama/Qwen model),
add multiple-choice distractors, OCR support, and automatic difficulty scoring.
