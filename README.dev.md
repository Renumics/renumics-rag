```shell
AZURE_RESOURCE_GROUP="VDISeminar"
AZURE_REGISTRY_NAME="${AZURE_RESOURCE_GROUP,,}acr"
AZURE_REGISTRY="${AZURE_REGISTRY_NAME}.azurecr.io"
```

**Create a new resource group:**

```shell
az group create --location germanywestcentral --resource-group "$AZURE_RESOURCE_GROUP"
```

**Create a new container registry:**

```shell
az acr create --resource-group "$AZURE_RESOURCE_GROUP" --sku Basic --location germanywestcentral --admin-enabled true --name "$AZURE_REGISTRY_NAME"
```

**Get username and password for the newly created container registry:**

```shell
AZURE_REGISTRY_CREDENTIALS="$(az acr credential show --name "$AZURE_REGISTRY_NAME")"
AZURE_REGISTRY_USERNAME="$(jq -n "$AZURE_REGISTRY_CREDENTIALS" | jq -r '.username')"
AZURE_REGISTRY_PASSWORD="$(jq -n "$AZURE_REGISTRY_CREDENTIALS" | jq -r '[.passwords | .[] | select(.name=="password")][0] | .value')"
```

**Log in to the newly created container registry:**

```shell
docker login -u "$AZURE_REGISTRY_USERNAME" -p "$AZURE_REGISTRY_PASSWORD" "$AZURE_REGISTRY"
```

**Build local image:**

```shell
docker build -t renumics-rag -f Dockerfile .
```

**Tag and push local image to registry, with timestamp and as latest:**

```shell
TIMESTAMP="$(shell date '+%Y-%m-%d_%H-%M-%S')"
docker tag renumics-rag "${AZURE_REGISTRY}/renumics-rag:$TIMESTAMP"
docker push "${AZURE_REGISTRY}/renumics-rag:$TIMESTAMP"
docker tag renumics-rag "${AZURE_REGISTRY}/renumics-rag:latest"
docker push "${AZURE_REGISTRY}/renumics-rag:latest"
```

**Create Azure app service plan with Linux and chosen subscription (1 CPU, 1.75 GB memory, 10 GB storage):**

```shell
az appservice plan create --resource-group "$AZURE_RESOURCE_GROUP" --location germanywestcentral --sku B1 --is-linux --name "$AZURE_RESOURCE_GROUP"
```

**Create environment:**

```shell
az containerapp env create --resource-group "$AZURE_RESOURCE_GROUP" --location germanywestcentral --name "${AZURE_RESOURCE_GROUP,,}-env"
```

**Create container app:**

```shell
az containerapp create --resource-group "$AZURE_RESOURCE_GROUP" --environment "${AZURE_RESOURCE_GROUP,,}-env" --allow-insecure false --image "${AZURE_REGISTRY}/renumics-rag" --registry-server "$AZURE_REGISTRY" --registry-username "$AZURE_REGISTRY_USERNAME" --registry-password "$AZURE_REGISTRY_PASSWORD" --secrets "openai-api-key=$OPENAI_API_KEY" --env-vars "OPENAI_API_KEY=secretref:openai-api-key" --ingress external --target-port 8000 --name "${AZURE_RESOURCE_GROUP,,}-1"
```
