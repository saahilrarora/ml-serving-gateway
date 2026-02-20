# use slim Python image to keep the container small
FROM python:3.11-slim

# set working directory inside the container
WORKDIR /app

# copy and install dependencies first (Docker caches this layer separately,
# so deps only reinstall when requirements.txt changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code and demo models
COPY gateway/ gateway/
COPY demo/ demo/

# expose the port uvicorn will listen on
EXPOSE 8000

# start the server — host 0.0.0.0 makes it accessible outside the container
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
