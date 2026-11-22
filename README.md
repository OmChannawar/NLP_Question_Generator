# 📚 QuizForge – Automatic Question Generation from Textbooks

QuizForge reads a whole textbook (**PDF or Word**), detects its **chapters and topics**, and generates exam-style
questions of several **types** at a chosen **difficulty** (Low / Mid / High) and **quantity** (5–30, step 5).
Questions can be **sorted and filtered by topic**, revealed with **answer keywords**, and **rated 1–5 stars**; the
rating comments are analysed with a Transformer **sentiment** model.

## 1. Key NLP / AI concepts

| Concept | Where it is used |
|---|---|
| **Seq2Seq** | Flan-T5 is an *encoder–decoder*: the encoder reads *instruction + answer + passage*, the decoder writes the question token by token (`qgen/generator.py`). |
| **Transformers** | Flan-T5 (question writing) and DistilBERT-SST-2 (sentiment) are Transformer models loaded via Hugging Face `transformers`. |
| **LLM** | Flan-T5 is instruction-tuned; difficulty and style are controlled by prompt wording (prompt engineering). |
| **Sentiment analysis** | DistilBERT fine-tuned on SST-2 classifies the users' comments on questions (`qgen/feedback.py`). |
| **Classical NLP** | Layout/heading detection, sentence splitting, TF-IDF sentence scoring, keyword extraction, TF-IDF similarity for wrong options. |

## 2. Features

0. **Soft (neumorphic) olive-and-cream UI** – raised cards/buttons and pressed-in inputs on a cream page, olive accents, brown
   text (palette `#556B2F` · `#FCF8D9` · `#5B4733`). Colours come from `.streamlit/config.toml`; the soft shadows from
   `assets/neumorphic.css`. A sidebar switch turns the soft skin off for a flat, higher-contrast look.

1. **Whole textbooks** – upload limit is 1000 MB (`.streamlit/config.toml`); pages are read one by one with live progress,
   running headers/footers/page numbers are removed.
2. **Chapters & topics** – detected from `Chapter N`, numbered sections (`3.1 …`), Word Heading styles and heading-like
   lines. Choose *Whole document* or only some chapters/topics before generating.
3. **Question types** (choose one or several; the count is split evenly):

   | Type | How it is made | How difficulty works |
   |---|---|---|
   | Subjective | Flan-T5 writes a question for a sentence chosen from the book | Low: factual (term) · Mid: explain/describe · High: reasoning sentence (why/how) |
   | MCQ | Flan-T5 writes the question for a key term; 3 wrong options from other terms in the book | Wrong options: Low = unrelated parts, Mid = same chapter, High = most similar terms |
   | Fill in the blanks | Textbook sentence with its key term blanked | High blanks two words |
   | True / False | Textbook sentence, or the same sentence with its term swapped for another term | Swapped term: far (low) … most similar (high) |
   | Match the pair | 4 terms ↔ 4 descriptions from definitions in the book | Low = far-apart terms, Mid = same chapter, High = similar descriptions |
4. **Sort by topic** – group by *Document order*, *Topic (A–Z)*, *Chapter → Topic* or *Question type*; filter by topic/type.
5. **Live progress** – every backend step is shown by name (e.g. *Reading document – page 120 of 480*) with a progress bar
   and a ✅/⏳/▫️ stage checklist.
6. **Answers** – *Answer the questions* shows keywords, the answer and its source sentence.
7. **Feedback + sentiment analysis** – rate each question's relevance to its topic (1–5 stars) and optionally comment.
   *Analyze feedback* classifies comments (positive / mixed / negative), shows average relevance, star distribution,
   relevance per topic (weakest first) and flags questions with ≤ 2 stars or a negative comment. CSV download included.

## 3. How it works

```
PDF/DOCX → pages → remove headers/footers → chapters/topics → passages (never cross a topic)
              │
   scope (all / chosen topics) → answer candidates (TF-IDF) → per type:
        Subjective, MCQ : Flan-T5 writes question → quality checks → (MCQ) add wrong options
        Fill/TrueFalse/Match : built directly from the sentences (no model, cannot hallucinate)
              │
   sort in book order → UI (group/filter) → ratings + comments → DistilBERT sentiment → summary
```
**Quality checks** for LLM questions: topic words must appear in the passage (no invented topics), the question must
not contain its own short answer, no near-duplicates; failed slots retry with another sentence of the same passage.
Answers always come from the book text, never from the model.

## 4. Tech stack

**Frontend** – Streamlit (`st.status`, `st.progress`, `st.feedback` star widget, charts), custom HTML/CSS.
**Backend** – Python 3.9+, PyTorch, Hugging Face Transformers + SentencePiece (Flan-T5 Small/Base/Large, DistilBERT-SST-2),
scikit-learn (TF-IDF), NumPy, pandas, pypdf, python-docx.

## 5. Project structure

```
question-generator/
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml  # upload limit (1000 MB) + colour theme
├── assets/
│   ├── base.css            # flat styles for QuizForge's own components
│   └── neumorphic.css      # soft-UI skin (shadows, inset inputs, Streamlit widget overrides)
├── notebooks/question_generation_demo.ipynb
└── qgen/
    ├── extractor.py        # PDF/DOCX reading, header/footer removal, chapter/topic detection, chunking
    ├── candidates.py       # answer selection + TermPool (wrong options)
    ├── qtypes.py           # fill-blank, true/false, match builders, MCQ options
    ├── generator.py        # Seq2Seq LLM prompts, batching, quality checks
    ├── pipeline.py         # orchestration with named progress stages
    ├── progress.py         # weighted stage progress tracker
    ├── feedback.py         # ratings + sentiment analysis + summary
    ├── formatting.py       # question/answer text for all types
    ├── keywords.py         # TF-IDF keywords
    └── schema.py           # shared data classes, types, validation
```

## 6. Installation & run

```bash
python -m venv venv
source venv/bin/activate          # Windows:  venv\Scripts\activate
pip install -r requirements.txt   # NVIDIA GPU? install a CUDA build of PyTorch first
streamlit run app.py              # run from the project folder so .streamlit/config.toml is used
```
First use downloads Flan-T5 (Base ≈ 1 GB) and, when comments are analysed, DistilBERT-SST-2 (≈ 270 MB).

## 7. Known limitations (please read)
* **Upload size** – Streamlit cannot be "unlimited"; the limit is set to 1000 MB. Very large files need RAM and time
  (the progress panel shows the reading speed). Scanned PDFs need OCR first.
* **Structure matters** – chapters/topics come from heading detection, which is heuristic. Books with clear headings work
  best; headless prose gets automatic keyword-based topic names. Headings can occasionally be missed or misjudged.
* **Question wording** – Flan-T5 is small; some Subjective/MCQ questions can be awkward. Use *Large* or *Better quality* mode.
* **Few definitions** – Match the pair and some MCQ/True-False questions need `Term: definition` / `X is a …` sentences;
  fewer questions than requested are returned (with a warning) when the text has too few.
* **Match / mixed-topic questions** – a match set is filed under the topic of its first term.
* **Sentiment** – the SST-2 model is English-only and has no neutral class (low-confidence results are shown as "mixed").
  Star ratings alone are ratings, not sentiment; sentiment analysis is applied to the *comments*.
  Feedback does not yet change future generation.
* **Neumorphic theme** – the soft look relies on CSS selectors of Streamlit's internal elements, which can change between
  Streamlit versions (it was checked on Streamlit 1.67 in Chromium). If a rule stops matching, that widget just keeps the
  plain olive/cream theme. Soft shadows have low contrast by design; use the sidebar switch for a flatter look. Pick the
  default/custom theme (not "Dark") in Streamlit's Settings menu.
* **Speed** – a GPU is strongly recommended; the sidebar shows whether one was detected.

## 8. Possible extensions
Regenerate flagged questions automatically, fine-tune T5 on SQuAD/RACE, QA-based answer verification, OCR, multilingual support.
