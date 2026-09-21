#!/usr/bin/env bash
# Provision a fresh Ubuntu VM to serve the API over HTTPS, permanently.
#
#   sudo bash provision-vm.sh <duckdns-subdomain> <duckdns-token>
#
# Written for an Oracle Cloud Always Free ARM instance (VM.Standard.A1.Flex, Ubuntu 22.04)
# but nothing here is Oracle-specific except the iptables note below.
#
# Why HTTPS is not optional: the front-end is served from Vercel over https, and a browser
# will not let an https page call a plain http API. Measured, with the browser's own words,
# in DEPLOY.md.
#
# Why DuckDNS rather than a purchased domain: Let's Encrypt issues certificates for names,
# not IP addresses, so a bare VM IP cannot have a real certificate. DuckDNS gives a free
# stable name; Caddy then gets and renews the certificate on its own.
set -euo pipefail

SUB="${1:?usage: provision-vm.sh <duckdns-subdomain> <duckdns-token>}"
TOKEN="${2:?usage: provision-vm.sh <duckdns-subdomain> <duckdns-token>}"
HOSTNAME="${SUB}.duckdns.org"
APP_DIR=/opt/radiology-hub

echo "==> provisioning ${HOSTNAME}"

# --- 1. Oracle's Ubuntu images ship a REJECT rule that blocks everything but SSH. --------
# The cloud-side security list is only half the story; miss this and ports 80/443 are open
# in the console and still refused at the host, which looks exactly like a dead service.
if command -v netfilter-persistent >/dev/null 2>&1; then
  iptables -I INPUT 5 -p tcp --dport 80  -j ACCEPT || true
  iptables -I INPUT 5 -p tcp --dport 443 -j ACCEPT || true
  netfilter-persistent save || true
  echo "    opened 80/443 in the host firewall"
fi

# --- 2. Docker --------------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  echo "    docker installed"
fi

# --- 3. Keep the DuckDNS record pointed at this machine ---------------------------------
# A free-tier public IP can change across a stop/start, and a stale record means a dead
# hostname with a valid certificate - the confusing kind of broken.
cat > /usr/local/bin/duckdns-update <<EOF
#!/bin/sh
curl -fsS "https://www.duckdns.org/update?domains=${SUB}&token=${TOKEN}&ip=" -o /var/log/duckdns.log
EOF
chmod +x /usr/local/bin/duckdns-update
/usr/local/bin/duckdns-update
( crontab -l 2>/dev/null | grep -v duckdns-update; echo "*/5 * * * * /usr/local/bin/duckdns-update" ) | crontab -
echo "    duckdns record set and refreshed every 5 min"

# --- 4. Build and run the API ------------------------------------------------------------
cd "$APP_DIR"
docker build -t radiology-hub-api .
docker rm -f radiology-api 2>/dev/null || true
docker run -d --name radiology-api --restart unless-stopped \
  -p 127.0.0.1:7860:7860 radiology-hub-api
echo "    api container up on 127.0.0.1:7860"

# --- 5. Caddy terminates TLS and proxies to it -------------------------------------------
# Bound to localhost above on purpose: the container is reachable only through Caddy, so
# there is no plain-http port open to the internet serving the same thing.
if ! command -v caddy >/dev/null 2>&1; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
fi

cat > /etc/caddy/Caddyfile <<EOF
${HOSTNAME} {
    reverse_proxy 127.0.0.1:7860

    # The page that calls this is on Vercel, so the answer has to say cross-origin is fine.
    # main.py already sends Access-Control-Allow-Origin: *; this is here so the header
    # survives if that ever changes, and so preflight is answered without waking the app.
    @preflight method OPTIONS
    respond @preflight 204

    header {
        Access-Control-Allow-Origin "*"
        Access-Control-Allow-Methods "GET, POST, OPTIONS"
        Access-Control-Allow-Headers "*"
        -Server
    }
}
EOF
systemctl reload caddy || systemctl restart caddy
echo "    caddy serving https://${HOSTNAME}"

echo
echo "==> done. Verify:"
echo "    curl https://${HOSTNAME}/health"
echo "    then point the page at it:"
echo "    https://ai-powered-radiology-hub.vercel.app/?api=https://${HOSTNAME}"
