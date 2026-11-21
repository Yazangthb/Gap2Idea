"""Local OpenAI-compatible adapter for the frozen five-paper pilot on Ollama.

Run against a private Ollama endpoint; explicitly disables thinking and fixes
the context size, neither of which the generic OpenAI endpoint guarantees.
"""
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA = os.environ.get('PILOT_OLLAMA_URL', 'http://127.0.0.1:11447').rstrip('/')
PORT = int(os.environ.get('PILOT_PROXY_PORT', '11448'))


def upstream(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode('utf8')
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Do not log prompts or credentials.

    def send_json(self, status, obj):
        raw = json.dumps(obj).encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            if self.path != '/v1/models':
                return self.send_json(404, {'error': 'unknown path'})
            models = upstream('/api/tags')['models']
            self.send_json(200, {'object': 'list', 'data': [
                {'id': m['name'], 'object': 'model', 'created': 0, 'owned_by': 'ollama'}
                for m in models]})
        except Exception as e:
            self.send_json(502, {'error': {'message': type(e).__name__}})

    def do_POST(self):
        try:
            if self.path != '/v1/chat/completions':
                return self.send_json(404, {'error': 'unknown path'})
            q = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            fmt = q['response_format']['json_schema']['schema']
            payload = {'model': q['model'], 'messages': q['messages'], 'stream': False,
                       'think': False, 'format': fmt,
                       'options': {'temperature': q['temperature'], 'seed': q['seed'],
                                   'num_predict': q['max_tokens'], 'num_ctx': 131072}}
            a = upstream('/api/chat', payload)
            if not a.get('done') or a['message'].get('thinking'):
                raise ValueError('Ollama did not finish or returned thinking')
            prompt, completion = a.get('prompt_eval_count', 0), a.get('eval_count', 0)
            self.send_json(200, {'id': 'ollama-pilot', 'object': 'chat.completion',
                'created': int(time.time()), 'model': a['model'],
                'choices': [{'index': 0, 'message': {'role': 'assistant',
                    'content': a['message']['content']}, 'finish_reason': a.get('done_reason', 'stop')}],
                'usage': {'prompt_tokens': prompt, 'completion_tokens': completion,
                          'total_tokens': prompt + completion}})
        except Exception as e:
            self.send_json(502, {'error': {'message': type(e).__name__}})


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
