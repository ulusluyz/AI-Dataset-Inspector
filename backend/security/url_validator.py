import ipaddress
import re
import socket
from urllib.parse import urlparse
from pydantic import BaseModel

class ParsedDatasetURL(BaseModel):
    raw_url: str
    normalized_url: str
    owner: str
    dataset_name: str
    dataset_id: str  # owner/dataset_name

class URLValidationError(Exception):
    pass

HF_HOSTNAMES = {"huggingface.co", "www.huggingface.co"}

# Private / Internal IP ranges
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / Cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),     # Multicast
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("::1/128"),         # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),        # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),       # IPv6 Link-local
]

def validate_and_parse_dataset_url(url: str) -> ParsedDatasetURL:
    """
    Strictly validates a Hugging Face dataset URL.
    Raises URLValidationError if the URL is invalid, unsafe, or not a dataset repository.
    """
    if not url or not isinstance(url, str):
        raise URLValidationError("Geçersiz bağlantı. URL boş olamaz.")

    url = url.strip()

    # Pre-parse check for scheme
    if not (url.startswith("https://") or url.startswith("http://")):
        raise URLValidationError("Geçersiz bağlantı. Yalnızca HTTP/HTTPS şemaları desteklenir.")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise URLValidationError(f"Geçersiz URL yapısı: {e}")

    # Enforce HTTPS/HTTP
    if parsed.scheme.lower() not in ("http", "https"):
        raise URLValidationError("Geçersiz bağlantı şeması. Yalnızca HTTP/HTTPS kabul edilir.")

    # Validate Hostname
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise URLValidationError("Geçersiz bağlantı. Sunucu adı (hostname) bulunamadı.")

    if hostname not in HF_HOSTNAMES:
        raise URLValidationError("Geçersiz bağlantı. Yalnızca Hugging Face (huggingface.co) dataset bağlantıları kabul edilir.")

    # Check for direct IP hostnames
    try:
        ip_obj = ipaddress.ip_address(hostname)
        for net in BLOCKED_IP_NETWORKS:
            if ip_obj in net:
                raise URLValidationError("Geçersiz bağlantı. Özel/yerel IP adreslerine erişim engellendi.")
    except ValueError:
        pass  # Hostname is a domain name, not an IP string

    # Path validation
    # Valid pattern: /datasets/<owner>/<dataset_name> or /datasets/<owner>/<dataset_name>/...
    path = parsed.path.strip("/")
    parts = [p for p in path.split("/") if p]

    if len(parts) < 3:
        raise URLValidationError(
            "Geçersiz bağlantı formatı. Dataset bağlantısı 'https://huggingface.co/datasets/<sahip>/<dataset_adı>' şeklinde olmalıdır."
        )

    if parts[0] != "datasets":
        if parts[0] in ("spaces", "models"):
            raise URLValidationError(f"Geçersiz bağlantı türü. Hugging Face {parts[0]} bağlantıları kabul edilmez, yalnızca dataset kabul edilir.")
        raise URLValidationError("Geçersiz bağlantı. Yalnızca Hugging Face dataset bağlantıları kabul edilir.")

    owner = parts[1]
    dataset_name = parts[2]

    # Validate owner and dataset name format (standard HF slug rule: alphanumeric, hyphens, underscores, dots)
    slug_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
    if not slug_pattern.match(owner) or not slug_pattern.match(dataset_name):
        raise URLValidationError("Geçersiz dataset sahibi veya adı formatı.")

    dataset_id = f"{owner}/{dataset_name}"
    normalized_url = f"https://huggingface.co/datasets/{dataset_id}"

    return ParsedDatasetURL(
        raw_url=url,
        normalized_url=normalized_url,
        owner=owner,
        dataset_name=dataset_name,
        dataset_id=dataset_id
    )

def check_ssrf_safe_host(hostname: str) -> bool:
    """
    DNS resolution safety check to ensure resolved IP does not fall into private/internal IP ranges.
    """
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip_obj = ipaddress.ip_address(ip_str)
            for net in BLOCKED_IP_NETWORKS:
                if ip_obj in net:
                    return False
        return True
    except Exception:
        return False
