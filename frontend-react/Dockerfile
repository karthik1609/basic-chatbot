FROM node:20-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
ARG NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=http://localhost:8000
ENV NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=${NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE}

COPY package*.json ./
RUN npm ci --no-audit --no-fund

COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE=${NEXT_PUBLIC_FASTAPI_EXTERNAL_BASE}

# Copy built app and runtime deps
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package*.json ./
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
# If you have a next.config.mjs, uncomment the next line to include it
# COPY --from=build /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm","run","start","--","-p","3000"]


