FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install -e .
COPY configs/ configs/
COPY prompts/ prompts/
EXPOSE 8501
CMD ["llm-eval", "run", "--split", "smoke"]
