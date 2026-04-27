$extra = "
  dashboard_pred:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: risk_dashboard_pred
    ports:
      - '8501:8501'
    depends_on:
      api_service:
        condition: service_healthy
    environment:
      - GIT_PYTHON_REFRESH=quiet
      - PYTHONPATH=/app/src
      - API_URL=http://risk_api:8000
    volumes:
      - ./src:/app/src:ro
      - ./data:/app/data:ro
      - ./artifacts:/app/artifacts:ro
    command: streamlit run src/credit_risk_analysis/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    restart: unless-stopped
