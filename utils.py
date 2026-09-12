import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

PHISHING_KEYWORDS = [
    "urgent", "verify", "verification", "account", "password", "login",
    "bank", "credit card", "payment", "confirm", "suspended", "suspend",
    "security alert", "unusual activity", "click here", "reset",
    "expire", "expired", "winner", "won", "prize", "gift card",
    "claim", "refund", "limited time", "immediately", "personal information",
    "social security", "otp", "one time password"
]

SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "cutt.ly", "rb.gy"
]

def extract_urls(text):
    return re.findall(r'https?://[^\s<>"\']+', str(text).lower())

def email_security_features(text):
    text = str(text)
    lower = text.lower()
    urls = extract_urls(text)

    url_count = len(urls)
    suspicious_url_count = 0
    ip_url_count = 0
    shortener_count = 0
    at_url_count = 0
    http_count = 0
    https_count = 0
    total_url_length = 0

    for url in urls:
        total_url_length += len(url)
        if "@" in url:
            at_url_count += 1
        if url.startswith("http://"):
            http_count += 1
        if url.startswith("https://"):
            https_count += 1
        if any(s in url for s in SHORTENERS):
            shortener_count += 1
        if re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', url):
            ip_url_count += 1
        if any(x in url for x in [".xyz", ".top", ".click", ".gq", ".tk", ".ml", ".ga"]):
            suspicious_url_count += 1

    keyword_count = sum(lower.count(k) for k in PHISHING_KEYWORDS)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    digit_count = sum(ch.isdigit() for ch in text)
    uppercase_count = sum(ch.isupper() for ch in text)
    text_length = len(text)
    word_count = len(re.findall(r"\b\w+\b", text))

    return [
        url_count,
        suspicious_url_count,
        ip_url_count,
        shortener_count,
        at_url_count,
        http_count,
        https_count,
        (total_url_length / url_count) if url_count else 0,
        keyword_count,
        exclamation_count,
        question_count,
        digit_count,
        uppercase_count,
        text_length,
        word_count,
    ]

class EmailSecurityFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [email_security_features(x) for x in X]
        return csr_matrix(np.asarray(rows, dtype=float))
