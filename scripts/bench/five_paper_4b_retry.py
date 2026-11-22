"""Follow-up to the failed 4B direct pilot: same five papers, clearer final instructions.

The original rubric and 8,192-token limit are unchanged. This is a new,
exploratory prompt, not a rerun of the frozen original configuration.
"""
import json
import time

import five_paper_pilot as pilot
from baseline_evidence import digest

SUFFIX = '''

END OF THE FIVE PAPERS.
Now return only the IDs of sentences that explicitly state a research gap,
limitation, or future-work direction, following the strict criteria above.
Do not list all IDs. Exclude ordinary methods, results, background, and titles.
The output must be one JSON object: {"positives":[ID, ...]}.
Include each qualifying ID once, in increasing order; if none qualify use [].

Example of the required selection (these are invented, not paper sentences):
[9001] We train a model on images.
[9002] Our method has not been tested on clinical data.
[9003] Accuracy improved by 3%.
For these three example sentences the answer is {"positives":[9002]}.
Do not include example IDs in your answer; use only the five papers' IDs.
'''


def main():
    plan = pilot.require_frozen()
    rows, _, _ = pilot.data()
    system, original, keys = pilot.group_prompt(rows)
    user = original + SUFFIX
    output = pilot.DEST / '4b_five_papers_prompt_retry.json'
    if output.exists():
        raise SystemExit('Retry output already exists; preserve it')
    c = pilot.client()
    served = 'qwen3.5-4b-bf16:latest'
    if served not in {x.id for x in c.models.list().data}:
        raise SystemExit('Expected model not available')
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['positives'],
              'properties': {'positives': {'type': 'array', 'items': {'type': 'integer'}}}}
    attempt = {}
    try:
        started = time.time()
        resp = c.chat.completions.create(model=served, temperature=0, max_tokens=pilot.MAX_OUTPUT,
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            response_format={'type': 'json_schema', 'json_schema': {'name': 'positives', 'schema': schema, 'strict': True}},
            extra_body={'seed': 0, 'chat_template_kwargs': {'enable_thinking': False}})
        attempt = {'seconds': round(time.time()-started, 3), 'finish_reason': resp.choices[0].finish_reason,
                   'usage': resp.usage.model_dump() if resp.usage else None, 'response': resp.choices[0].message.content}
        if attempt['finish_reason'] != 'stop':
            raise ValueError('Response did not finish within the output limit')
        parsed = json.loads(attempt['response']); ids = parsed.get('positives')
        if (set(parsed) != {'positives'} or not isinstance(ids, list) or
            any(type(i) != int or not 0 <= i < len(keys) for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError('Invalid, duplicate or out-of-range IDs')
        kept = {tuple(keys[i]) for i in ids}
        result = {'status': 'ok', 'model': served, 'plan_sha256': pilot.sha(pilot.PLAN),
                  'prompt_sha256': digest(user), 'suffix': SUFFIX, 'attempt': attempt,
                  'kept': [list(k) for k in sorted(kept)], 'metrics': pilot.metrics(rows, kept)}
    except Exception as exc:
        result = {'status': 'failed', 'model': served, 'plan_sha256': pilot.sha(pilot.PLAN),
                  'prompt_sha256': digest(user), 'suffix': SUFFIX, 'attempt': attempt,
                  'error_type': type(exc).__name__, 'error': str(exc)[:200]}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'status': result['status'], 'metrics': result.get('metrics'),
                      'finish_reason': attempt.get('finish_reason'), 'usage': attempt.get('usage')}), flush=True)


if __name__ == '__main__':
    main()
