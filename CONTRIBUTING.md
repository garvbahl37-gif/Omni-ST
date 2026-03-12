# Contributing to Omni-ST

We welcome all contributions — from bug fixes to new modalities to GSoC proposals!

## 🏁 Getting Started

```bash
git clone https://github.com/omni-st/omni-st.git
cd omni-st
pip install -e ".[dev]"
pre-commit install
```

## 🧩 Adding a New Modality

1. Create `models/your_modality_encoder.py` implementing `forward(inputs) → Tensor[B, output_dim]`
2. Add a projector entry in `MultimodalFusionBackbone.modality_projectors`
3. Add `MODALITY_IDS["your_modality"] = N` in `multimodal_backbone.py`
4. Update `configs/model/default.yaml` with your encoder config
5. Add tests in `tests/test_your_modality.py`

## 🎯 Adding a New Task

1. Create a task head in `tasks/task_heads.py`
2. Add instruction template to `models/text_encoder.py` `INSTRUCTION_TEMPLATES`
3. Implement dataset logic in the appropriate dataset class
4. Add metrics to `evaluation/metrics.py`

## 💡 GSoC Project Ideas

| Project | Difficulty | Labels |
|---|---|---|
| Xenium / MERFISH dataset loader | Medium | `gsoc`, `datasets` |
| 3-D multi-section spatial modelling | Hard | `gsoc`, `architecture` |
| Pre-trained model weights & HF Hub integration | Medium | `gsoc`, `mlops` |
| Interactive web demo (Streamlit/Gradio) | Easy | `gsoc`, `visualization` |
| Protein expression (CITE-seq) modality | Hard | `gsoc`, `multimodal` |

## 📝 Code Style

- `black` (100 char line length)
- `isort` (profile: black)
- `flake8` (max-line-length 100)
- Type hints required for all public functions
- Docstrings in NumPy format

## 🧪 Tests

```bash
pytest tests/ -v --cov=models --cov-report=term-missing
```

## 📬 Pull Request Guidelines

- Fork → feature branch → PR to `main`
- PR must include: description, tests, updated docs
- Ensure CI passes before requesting review
