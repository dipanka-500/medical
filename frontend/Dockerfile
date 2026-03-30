# ═══════════════════════════════════════════════════════════════════
#  MedAI Frontend — Production Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Build ───────────────────────────────────────────────
FROM node:20-alpine AS build

ARG VITE_API_URL=/api/v1

WORKDIR /app

# Install deps first (cache layer)
COPY package.json package-lock.json* yarn.lock* ./
RUN npm ci --prefer-offline --no-audit --no-fund 2>/dev/null || npm install --prefer-offline --no-audit --no-fund

# Build app
COPY . .
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ── Stage 2: Serve (hardened Nginx) ──────────────────────────────
FROM nginx:1.27-alpine AS runtime

ARG BUILD_DATE

LABEL maintainer="MedAI Platform Team" \
      org.opencontainers.image.title="medai-frontend" \
      org.opencontainers.image.description="MedAI Platform Frontend" \
      org.opencontainers.image.created="${BUILD_DATE}"

# Remove default config and server tokens
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf && \
    sed -i 's/worker_processes  1/worker_processes auto/' /etc/nginx/nginx.conf

# Copy build output
COPY --from=build /app/dist /usr/share/nginx/html

# Copy hardened nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Non-root nginx (alpine nginx already supports this)
RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /var/log/nginx && \
    touch /var/run/nginx.pid && \
    chown -R nginx:nginx /var/run/nginx.pid

USER nginx

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost:3000/ || exit 1

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
