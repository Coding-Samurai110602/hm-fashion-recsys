# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:22-alpine AS builder
WORKDIR /app

# Install deps first (better layer caching)
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts

# Build the SPA
COPY . .
RUN npm run build

# ── Stage 2: serve ────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Redirect nginx pid to /tmp so nginx can run as non-root.
# nginx:1.27-alpine uses /run/nginx.pid (earlier versions used /var/run/nginx.pid).
RUN sed -i 's|/run/nginx.pid|/tmp/nginx.pid|g' /etc/nginx/nginx.conf \
    && mkdir -p /tmp \
    && touch /tmp/nginx.pid \
    && chown -R nginx:nginx \
        /usr/share/nginx/html \
        /var/cache/nginx \
        /var/log/nginx \
        /etc/nginx/conf.d \
        /tmp/nginx.pid

COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html
COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf

USER nginx
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO /dev/null http://localhost:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
