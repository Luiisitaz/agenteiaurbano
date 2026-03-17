# Urban AI Agent - Reportes Urbanos Inteligentes

**Para probar el agente entra aqui:** [http://35.226.107.16:8000/app](http://35.226.107.16:8000/app)


Este proyecto implementa un asistente ciudadano para reportar incidentes urbanos usando LangGraph, FastAPI, PostgreSQL + PostGIS + pgvector y un mapa Leaflet.

## Requisitos
- Python 3.10+
- Docker + Docker Compose

## Variables de entorno (.env)
Crea o edita el archivo `.env` en la raiz del proyecto.

- `OPENAI_API_KEY`: API key de OpenAI (requerida para chat y embeddings).
- `OPENAI_BASE_URL`: URL base de la API (opcional, por defecto `https://api.openai.com/v1`).
- `GOOGLE_MAPS_API_KEY`: API key de Google Maps (requerida si usas geocoding con Google Places).
- `GEOCODING_PROVIDER`: `google_places` o `nominatim` (opcional).
- `GOOGLE_PLACES_REGION`: region de sesgo para Google Places (opcional, default `pa`).
- `GOOGLE_PLACES_LANGUAGE`: idioma para Google Places (opcional, default `es`).
- `CLUSTER_RADIUS_M`: radio fijo para clustering de incidentes en metros (opcional, default `100`).
- `DATABASE_URL`: URL de Postgres (opcional). Default: `postgresql+psycopg2://postgres:12345@localhost:5433/urban_reports`.
- `LOG_LEVEL`: nivel de logs (opcional, default `info`).

Ejemplo:
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
GOOGLE_MAPS_API_KEY=AIza...
GEOCODING_PROVIDER=google_places
GOOGLE_PLACES_REGION=pa
GOOGLE_PLACES_LANGUAGE=es
CLUSTER_RADIUS_M=100
DATABASE_URL=postgresql+psycopg2://postgres:12345@localhost:5433/urban_reports
LOG_LEVEL=info
```

Los logs del agente salen por `stdout` en tiempo real. No se escriben a archivo.
Si `GOOGLE_MAPS_API_KEY` esta configurada, el agente prioriza Google Places para resolver nombres de lugares y puede pedir confirmacion de la mejor coincidencia antes de crear el reporte.

## Configuracion del modelo (config.ini)
Archivo `config.ini` en la raiz. Aqui eliges modelos y parametros (no secretos).

Secciones clave:
- `[openai]`
  - `base_url`: URL base de la API.
  - `chat_model`: modelo principal (default `gpt-5.4`).
  - `chat_model_fallback`: modelo alternativo (default `gpt-5-mini`).
  - `embedding_model`: default `text-embedding-3-small`.
  - `embedding_dimensions`: default `1536`.
- `[app]`
  - `duplicate_threshold`: umbral de duplicados (0-1).
  - `nearby_radius_m`: radio de busqueda (metros).
  - `cluster_radius_m`: radio fijo de clustering (metros).
  - `max_results`: maximo de resultados.
- `[geocoding]`
  - `provider`: `nominatim` o `google_places`.
  - `user_agent`: user-agent para Nominatim.
  - `google_region`: region para Google Places (default `pa`).
  - `google_language`: idioma para Google Places (default `es`).

> Nota: si cambias `embedding_dimensions`, debes re-crear la columna `embedding` o re-inicializar la base de datos.

## Base de datos (Docker)
Se usa una imagen Postgres con PostGIS + pgvector.

1. Levanta el contenedor:
```
docker compose up -d db
```

2. Inicializa extensiones y esquema:
```
# Windows (PowerShell)
./scripts/init_db.ps1

# Linux/Mac
./scripts/init_db.sh
```

## Instalacion de dependencias
```
python -m pip install -r requirements.txt
```

## Ejecutar API
```
python -m uvicorn app.main:app --reload
```

## Ejecutar Todo En Docker
```
docker compose up -d --build
docker compose logs -f app
```

## Despliegue en servidor (VM) con Docker Compose
> Recomendado para producción o pruebas remotas.

### Opción A: Paso a paso manual
1. Conéctate por SSH a tu VM.
2. Instala Docker + docker-compose:
```
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose git
sudo systemctl enable --now docker
```
3. Clona el repositorio:
```
sudo mkdir -p /opt/agenteia
sudo chown -R $USER:$USER /opt/agenteia
cd /opt/agenteia
git clone https://github.com/Luiisitaz/agenteia.git
cd agenteia
```
4. Crea el `.env` en el servidor:
```
cat > .env <<'EOF'
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
GOOGLE_MAPS_API_KEY=AIza...
LOG_LEVEL=info
EOF
```
5. Levanta los servicios:
```
sudo docker-compose up -d --build
sudo docker-compose ps
```
6. Abre el puerto **8000** en el firewall de tu proveedor cloud.
7. Accede:
```
http://<SERVER_IP>:8000/app
```

### Opción B: Script automático
En la VM ejecuta:
```
chmod +x scripts/deploy_vm.sh
./scripts/deploy_vm.sh
```
El script:
- Instala Docker + docker-compose
- Clona el repo en `/opt/agenteia`
- Crea `.env` si hay variables en el entorno
- Levanta los contenedores

> Si no tienes variables exportadas, el script crea un `.env` de plantilla y te pide editarlo.

Endpoints:
- `POST /chat`
- `GET /reports/clusters`
- `GET /reports/map`
- `GET /reports/{id}`
- `GET /map` (mapa Leaflet simple)
- `GET /app` (dashboard completo)

## Ejemplos de uso
### Crear reporte
```
POST /chat
{
  "message": "Hay una fuga de agua en Via Espana frente a la Delta. Mi cedula es 8-123-4567"
}
```

### Reportes cercanos
```
POST /chat
{
  "message": "Hay reportes de baches cerca de Via Espana?"
}
```

### Mis reportes
```
POST /chat
{
  "message": "Mis reportes con cedula 8-123-4567"
}
```

## Estructura del proyecto
```
app/
  agent.py
  config.py
  database.py
  embeddings.py
  main.py
  models.py
  rag.py
  schemas.py
  tools.py
  static/map.html
migrations/init.sql
scripts/init_db.py
scripts/init_db.ps1
scripts/init_db.sh
config.ini
requirements.txt
docker-compose.yml
```
