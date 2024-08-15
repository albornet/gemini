# Medical LLM Inference

A Python script to prompt a large language model (LLM) for predicting the Modified Rankin Scale (mRS) score from medical text.

## Usage

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows use `venv\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your dataset csv file(s) at `data/`

3. Run the script:
   ```bash
   python inference.py
   ```

4. Find the results at `results/`.
