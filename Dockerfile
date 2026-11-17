# API del limitador: docker build -t fcl-api . && docker run -p 8000:8000 fcl-api
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy scipy fastapi "uvicorn[standard]" pydantic
COPY fcl/ fcl/
COPY api/ api/
COPY modelos/ modelos/
ENV FCL_DB=/app/datos/disenos.sqlite
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
