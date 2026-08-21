## tos-to-odrl — Automated Terms of Service to TOSL Pipeline

Usage examples:
- Standard OpenAI (reads OPENAI_API_KEY from .env):
    ```
    python main.py --input data/elsevier/api_service_agreement_2017.txt \
    --provider "Elsevier" --title "API Service Agreement" --date "2017" \
    --output output/elsevier-2017/
    ```
    
- Pre-structured JSON (clauses already split skips Step 0):
    ```
    python main.py --input data/use_cases_elsevier.json --output output/
    ```

- Custom OpenAI-compatible server (vLLM, Ollama, LM Studio, etc.):
    ```
    python main.py --input tos.txt --base-url http://localhost:8000/v1 --model llama3 \
    --output output/
    ```

- Custom server with explicit API key:
    ```
    python main.py --input tos.txt --base-url http://my-server/v1 --api-key my-key \
    --model mistral --output output/
    ```
