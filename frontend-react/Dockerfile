FROM node:20-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=http://localhost:8000
ENV NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=${NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE}

COPY package*.json ./
# Prefer reproducible ci install; fall back to npm install when lock is out of sync
RUN set -e; \
  if npm ci --no-audit --no-fund; then \
    echo "npm ci succeeded"; \
  else \
    echo "npm ci failed; falling back to npm install"; \
    npm install --no-audit --no-fund; \
  fi

COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=${NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE}

# Copy built app and runtime deps
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/package*.json ./
COPY --from=build --chown=node:node /app/.next ./.next
COPY --from=build --chown=node:node /app/public ./public
# If you have a next.config.mjs, uncomment the next line to include it
# COPY --from=build --chown=node:node /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
USER node
CMD ["npm","run","start","--","-p","3000"]


